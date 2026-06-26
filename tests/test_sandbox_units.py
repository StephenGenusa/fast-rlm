"""sandbox.py offline units: env scrub, cache dir, recv/await timeouts, hardening
flags (Popen mocked), pickle helper, and SandboxPool internals."""
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rlm.sandbox as S                                   # noqa: E402
from rlm.sandbox import PyodideSandbox, SandboxPool, _b64pickle   # noqa: E402

HAVE_RUNTIME = shutil.which("deno") or shutil.which("node")


def test_b64pickle_roundtrips():
    import base64, pickle
    obj = {"a": 1, "b": [1, 2, {3}]}
    assert pickle.loads(base64.b64decode(_b64pickle(obj))) == obj


def test_child_env_scrubs_keys_keeps_essentials():
    if not HAVE_RUNTIME:
        print("SKIP (no runtime)"); return
    os.environ["ANTHROPIC_API_KEY"] = "sk-secret"
    os.environ["OPENAI_API_KEY"] = "sk-other"
    sb = PyodideSandbox(allow_unsafe_node=True)
    try:
        env = sb._child_env()
        assert "ANTHROPIC_API_KEY" not in env and "OPENAI_API_KEY" not in env
        assert "PATH" in env
    finally:
        sb.close()


def test_deno_cache_dir_from_env():
    if not HAVE_RUNTIME:
        print("SKIP (no runtime)"); return
    sb = PyodideSandbox(allow_unsafe_node=True)
    try:
        os.environ["DENO_DIR"] = "/tmp/some_cache"
        assert sb._deno_cache_dir() == "/tmp/some_cache"
    finally:
        sb.close()


def test_recv_and_await_event_time_out():
    if not HAVE_RUNTIME:
        print("SKIP (no runtime)"); return
    sb = PyodideSandbox(allow_unsafe_node=True)
    try:
        sb._q = queue.Queue()
        t0 = time.time()
        try:
            sb._recv(timeout=0.05); assert False
        except TimeoutError:
            pass
        try:
            sb._await_event("ready", 0.2); assert False
        except TimeoutError:
            assert 0.15 < time.time() - t0 < 2.0
    finally:
        sb.close()


class _FakeStdin:
    def write(self, s): pass
    def flush(self): pass


class _FakeStdout:
    def __init__(self): self._lines = ['{"t":"ready"}\n']
    def readline(self): return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    def __init__(self): self.stdin = _FakeStdin(); self.stdout = _FakeStdout()
    def poll(self): return None
    def wait(self, timeout=None): return 0
    def kill(self): pass


def test_deno_start_uses_hardening_flags():
    if not shutil.which("deno"):
        print("SKIP (no deno)"); return
    os.environ["DENO_DIR"] = "/tmp/deno_cache_x"
    captured = {}
    orig = S.subprocess.Popen
    S.subprocess.Popen = lambda cmd, **kw: (captured.update(cmd=cmd, kw=kw) or _FakeProc())
    try:
        sb = PyodideSandbox(runtime="deno")
        sb.start()
        cmd = captured["cmd"]
        assert "--deny-env" in cmd and "--deny-net" in cmd and "--deny-write" in cmd
        assert "--deny-run" in cmd and "--deny-ffi" in cmd
        assert "--node-modules-dir=none" in cmd
        assert any(c.startswith("--allow-read=/tmp/deno_cache_x") for c in cmd)
        assert "ANTHROPIC_API_KEY" not in captured["kw"].get("env", {})   # scrubbed env
    finally:
        S.subprocess.Popen = orig


# -- SandboxPool internals --------------------------------------------------
class _Stub:
    def __init__(self): self.reset_calls = 0; self.closed = False
    def start(self): pass
    def reset(self): self.reset_calls += 1
    def close(self): self.closed = True


def test_pool_alive_variants():
    import types
    pool = SandboxPool(factory=_Stub, size=1)
    assert pool._alive(_Stub()) is True                                   # no _proc -> usable
    assert pool._alive(types.SimpleNamespace(_proc=None)) is False        # dead
    assert pool._alive(types.SimpleNamespace(_proc=types.SimpleNamespace(poll=lambda: None))) is True
    assert pool._alive(types.SimpleNamespace(_proc=types.SimpleNamespace(poll=lambda: 0))) is False


def test_pool_lease_context_manager_releases():
    pool = SandboxPool(factory=_Stub, size=1)
    with pool.lease() as sb:
        assert isinstance(sb, _Stub)
    assert pool.acquire() is sb and sb.reset_calls >= 1   # released + reset, reused


def test_pool_close_drains_free():
    pool = SandboxPool(factory=_Stub, size=2)
    a = pool.acquire(); pool.release(a)
    pool.close()
    assert a.closed is True


def test_pool_blocks_until_release():
    pool = SandboxPool(factory=_Stub, size=1)
    a = pool.acquire()
    got = {}
    def worker(): got["sb"] = pool.acquire(timeout=2)
    th = threading.Thread(target=worker); th.start()
    time.sleep(0.2)
    assert "sb" not in got            # blocked (pool empty, at cap)
    pool.release(a)
    th.join(timeout=2)
    assert got.get("sb") is a         # served the released one
    pool.close()


def test_pool_factory_error_decrements_count():
    boom = {"n": 0}
    def factory():
        boom["n"] += 1
        raise RuntimeError("nope")
    pool = SandboxPool(factory=factory, size=1)
    try:
        pool.acquire(); assert False
    except RuntimeError:
        pass
    assert pool._count == 0           # rolled back so the pool isn't permanently full


def test_run_recovers_from_midrun_crash():
    if not HAVE_RUNTIME:
        print("SKIP (no runtime)"); return
    sb = PyodideSandbox(allow_unsafe_node=True)
    try:
        sb._send = lambda obj: None
        sb._kill = lambda: None
        sb._restart = lambda: None
        def boom(timeout=None): raise RuntimeError("sandbox process exited unexpectedly")
        sb._recv = boom
        r = sb.run("code", lambda *a: None)
        assert not r.has_final and "SandboxCrash" in r.stdout
    finally:
        sb.close()


def test_pool_acquire_timeout_raises_timeouterror():
    pool = SandboxPool(factory=_Stub, size=1)
    pool.acquire()                      # leased; none free, at cap
    try:
        pool.acquire(timeout=0.1); assert False
    except TimeoutError:
        pass
    pool.close()


def test_pool_close_closes_leased_sandboxes():
    pool = SandboxPool(factory=_Stub, size=1)
    a = pool.acquire()                  # checked out, never released
    pool.close()
    assert a.closed is True             # leased sandbox is reclaimed on close


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa
                failures += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("\n" + ("ALL TESTS PASSED" if not failures else f"{failures} FAILURE(S)"))
    sys.exit(1 if failures else 0)
