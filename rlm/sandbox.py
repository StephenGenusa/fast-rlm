"""
Sandboxes for executing model-generated code.

Two backends, one interface:

- ``PyodideSandbox`` — runs code as CPython-in-WASM inside a Deno (preferred) or
  Node subprocess. The code has no network and no API keys; its only exit is the
  ``llm_query`` / ``batch_llm_query`` channel, which the host fully controls.
  Under Deno the runtime layer denies net/write/run/ffi **and** env, scopes
  filesystem read to the Deno cache dir (+ a private temp cwd), and runs in a
  throwaway working directory — so even a WASM escape cannot read host files or
  secrets, reach the network, write outside the temp dir, or spawn. The subprocess
  environment is scrubbed of API keys on both runtimes (the host makes the real
  LLM calls). A per-cell compute deadline kills+restarts a runaway interpreter.
  **Node has no OS permission layer; using it requires allow_unsafe_node=True.**

The interface:

    sb = PyodideSandbox(); sb.start(); sb.set_context(context)
    result = sb.run(code, sub_handler)   # -> SandboxResult
    sb.close()
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

_HOST_SCRIPT = str(Path(__file__).with_name("sandbox_host.mjs"))

SubHandler = Callable[[str, Any], Any]


def _b64pickle(obj: Any) -> str:
    import base64
    import pickle
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


@dataclass
class SandboxResult:
    stdout: str
    final: Any = None
    has_final: bool = False


class Sandbox(Protocol):
    async_subcalls: bool
    def start(self) -> None: ...
    def set_context(self, context: Any) -> None: ...
    def run(self, code: str, sub_handler: SubHandler) -> SandboxResult: ...
    def reset(self) -> None: ...
    def clone(self) -> "Sandbox": ...
    def close(self) -> None: ...


class PyodideSandbox:
    """Run model code in Pyodide inside a locked-down Deno/Node subprocess."""

    async_subcalls = True

    # Deny everything dangerous; read is scoped (see start()).
    DENO_DENY = ["--deny-env", "--deny-net", "--deny-write", "--deny-run", "--deny-ffi"]

    # Env the child runtime legitimately needs; everything else (notably the
    # provider API keys) is scrubbed so the sandbox process never holds a secret.
    _ENV_KEEP = ("PATH", "HOME", "DENO_DIR", "NODE_PATH", "TMPDIR", "TEMP", "TMP",
                 "LANG", "LC_ALL", "SYSTEMROOT", "SystemRoot", "USERPROFILE", "APPDATA")

    def __init__(self, runtime: str = "auto", index_url: str | None = None,
                 stdout_head: int = 4000, stdout_tail: int = 1000,
                 startup_timeout: float = 120.0, exec_timeout: float | None = 120.0,
                 allow_unsafe_node: bool = False, context_codec: str = "auto"):
        self.allow_unsafe_node = allow_unsafe_node
        self.context_codec = context_codec
        self.runtime = self._pick_runtime(runtime)
        if self.runtime == "node" and not allow_unsafe_node:
            raise RuntimeError(
                "The Node runtime has no OS-level permission layer, so model code is "
                "contained only by the WASM boundary (no scoped read / denied env / "
                "denied net at the process level). Install Deno for a hardened sandbox, "
                "or pass allow_unsafe_node=True to accept the weaker isolation "
                "(trusted input only).")
        self.index_url = index_url
        self.head = stdout_head
        self.tail = stdout_tail
        self.startup_timeout = startup_timeout
        self.exec_timeout = exec_timeout
        self._proc: subprocess.Popen | None = None
        self._run_seq = 0
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._reader: threading.Thread | None = None
        self._cwd = tempfile.mkdtemp(prefix="rlm-sandbox-")
        self._context: Any = None
        self._has_context = False

    # -- locked-down child environment --------------------------------------
    def _child_env(self) -> dict:
        return {k: os.environ[k] for k in self._ENV_KEEP if k in os.environ}

    def _deno_cache_dir(self) -> str | None:
        d = os.environ.get("DENO_DIR")
        if d:
            return d
        try:
            out = subprocess.check_output(
                ["deno", "info", "--json"], text=True,
                stderr=subprocess.DEVNULL, env=self._child_env())
            return json.loads(out).get("denoDir")
        except Exception:  # noqa
            return None

    def _pick_runtime(self, runtime: str) -> str:
        if runtime != "auto":
            if not shutil.which(runtime):
                raise RuntimeError(f"requested runtime {runtime!r} not found on PATH")
            return runtime
        for cand in ("deno", "node"):
            if shutil.which(cand):
                return cand
        raise RuntimeError(
            "no JS runtime found. Install Deno (recommended) or Node, plus the "
            "pyodide npm package, to use the Pyodide sandbox.")

    def _node_index_url(self) -> str:
        if self.index_url:
            return self.index_url
        try:
            out = subprocess.check_output(
                ["node", "-e",
                 "const p=require('path');process.stdout.write(p.dirname(require.resolve('pyodide')))"],
                text=True, stderr=subprocess.DEVNULL)
            return out.strip()
        except Exception as e:  # noqa
            raise RuntimeError(
                "could not locate the 'pyodide' npm package. Run `npm install pyodide` "
                "in your working directory (or pass index_url=...).") from e

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return  # already running — start() is idempotent
        self._q = queue.Queue()  # fresh queue (supports restart)
        if self.runtime == "deno":
            cache = self._deno_cache_dir()
            reads = [p for p in (cache, self._cwd) if p]
            if cache:
                read_flag = "--allow-read=" + ",".join(reads)
            else:
                read_flag = "--allow-read"
                sys.stderr.write(
                    "[rlm] WARNING: could not resolve the Deno cache dir; falling back "
                    "to unrestricted --allow-read. Set DENO_DIR to scope filesystem reads.\n")
            cmd = ["deno", "run", "--node-modules-dir=none", read_flag,
                   *self.DENO_DENY, _HOST_SCRIPT]
            if self.index_url:
                cmd.append(self.index_url)
        else:
            sys.stderr.write(
                "[rlm] WARNING: using Node for the Pyodide sandbox (allow_unsafe_node). "
                "No OS-level permission layer; containment is only the WASM boundary.\n")
            cmd = ["node", _HOST_SCRIPT, self._node_index_url()]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, bufsize=1, env=self._child_env(), cwd=self._cwd)
        self._reader = threading.Thread(
            target=self._read_loop, args=(self._proc, self._q), daemon=True)
        self._reader.start()
        try:
            self._await_event("ready", self.startup_timeout)
        except (TimeoutError, RuntimeError) as e:
            self._kill()
            raise RuntimeError(
                f"sandbox failed to start ({e}). If this is the Deno runtime with no "
                "network, pre-cache Pyodide first: `deno cache npm:pyodide` (or run "
                "scripts/prefetch.sh). For Node: `npm install pyodide`.") from e

    # Contexts at or below this serialized size go in one `init` message (simple,
    # back-compat); larger ones are streamed in chunks so a multi-MB context never
    # has to cross stdio as a single giant JSON line.
    _INIT_INLINE_MAX = 256_000
    _CHUNK = 1_000_000

    def set_context(self, context: Any) -> None:
        self._context = context
        self._has_context = True
        kind, payload = self._encode_context(context)
        if len(payload) <= self._INIT_INLINE_MAX:
            self._send({"t": "init", "payload": payload, "contextKind": kind,
                        "head": self.head, "tail": self.tail})
            self._check_inited(self._await_event("inited", 60.0))
            return
        # streamed path
        self._send({"t": "init_begin", "contextKind": kind,
                    "head": self.head, "tail": self.tail})
        for i in range(0, len(payload), self._CHUNK):
            self._send({"t": "ctx_chunk", "data": payload[i:i + self._CHUNK]})
        self._send({"t": "init_end"})
        # assembly + (for json/pickle) decode scales with size; give it room
        timeout = max(60.0, len(payload) / 1_000_000 * 15.0)
        self._check_inited(self._await_event("inited", timeout))

    def _encode_context(self, context: Any) -> tuple[str, str]:
        """Return (contextKind, wire-string). str -> raw; JSON-able -> json;
        anything else -> base64(pickle) so arbitrary objects can cross the
        process/WASM boundary (the object's classes must exist in the sandbox)."""
        codec = self.context_codec
        if codec == "pickle":
            return "pickle", _b64pickle(context)
        if isinstance(context, str):
            return "str", context
        if codec in ("auto", "json"):
            try:
                return "json", json.dumps(context)
            except (TypeError, ValueError):
                if codec == "json":
                    raise TypeError(
                        "context is not JSON-serializable; use context_codec='auto' "
                        "or 'pickle'")
                return "pickle", _b64pickle(context)
        raise ValueError(f"unknown context_codec {codec!r}")

    @staticmethod
    def _check_inited(msg) -> None:
        if isinstance(msg, dict) and msg.get("error"):
            raise RuntimeError(
                "context could not be bound in the sandbox: " + str(msg["error"]) +
                ". With the pickle codec the object's classes must be importable "
                "inside Pyodide — load their module/source into the sandbox, or pass "
                "a JSON-serializable context.")

    def reset(self) -> None:
        """Clear the REPL namespace AND unbind the context (cheap; no re-send).

        Used by SandboxPool between leases so a warm subprocess carries no state
        (or large prior context) into the next query."""
        if self._proc is None or self._proc.poll() is not None:
            return
        self._send({"t": "reset"})
        self._await_event("reset_done", 30.0)
        self._has_context = False
        self._context = None

    def clone(self) -> "PyodideSandbox":
        """A fresh, unstarted sandbox with the same configuration."""
        return PyodideSandbox(
            runtime=self.runtime, index_url=self.index_url,
            stdout_head=self.head, stdout_tail=self.tail,
            startup_timeout=self.startup_timeout, exec_timeout=self.exec_timeout,
            allow_unsafe_node=self.allow_unsafe_node)

    def run(self, code: str, sub_handler: SubHandler) -> SandboxResult:
        self._run_seq += 1
        rid = f"r{self._run_seq}"
        self._send({"t": "run", "id": rid, "code": code})
        remaining = self.exec_timeout
        while True:
            t0 = time.monotonic()
            try:
                msg = self._recv(timeout=remaining)
            except (TimeoutError, RuntimeError) as e:
                # TimeoutError: runaway WASM compute (e.g. `while True`) that can't be
                # interrupted cooperatively. RuntimeError: the subprocess exited mid-run
                # (crash/OOM). Either way: kill, restart, and surface a graceful result.
                reason = (f"ExecutionTimeout: cell exceeded {self.exec_timeout}s of sandbox compute"
                          if isinstance(e, TimeoutError)
                          else f"SandboxCrash: the sandbox process exited mid-run ({e})")
                self._kill()
                try:
                    self._restart()
                except Exception as re:  # noqa
                    return SandboxResult(f"{reason}; sandbox restart failed: {re}", None, False)
                return SandboxResult(
                    f"{reason}; the sandbox was reset (REPL variables cleared).", None, False)
            if remaining is not None:
                remaining = max(0.0, remaining - (time.monotonic() - t0))
            t = msg.get("t")
            if t == "sub":
                try:
                    value = sub_handler(msg["kind"], msg["payload"])
                except Exception as e:  # surface sub-call failures into the REPL
                    value = f"[sub-call error: {type(e).__name__}: {e}]"
                self._send({"t": "sub_result", "id": msg["id"], "value": value})
            elif t == "result" and msg.get("id") == rid:
                return SandboxResult(msg["stdout"], msg.get("final"),
                                     msg.get("hasFinal", False))

    def _kill(self) -> None:
        if self._proc:
            try:
                self._proc.kill()
            except Exception:  # noqa
                pass
        self._proc = None

    def _restart(self) -> None:
        self.start()
        if self._has_context:
            self.set_context(self._context)

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send({"t": "close"})
                self._proc.wait(timeout=5)
            except Exception:  # noqa
                self._proc.kill()
        self._proc = None
        shutil.rmtree(self._cwd, ignore_errors=True)

    # -- low-level stdio ----------------------------------------------------
    def _read_loop(self, proc, q) -> None:
        # Bind to the proc/queue this thread was started for, so a stale reader
        # from a killed process never drops its sentinel into a fresh queue.
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                q.put(line)
        finally:
            q.put(None)

    def _send(self, obj: dict) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout: float | None = None) -> dict:
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"no message from sandbox within {timeout}s")
        if line is None:
            raise RuntimeError("sandbox process exited unexpectedly")
        return json.loads(line)

    def _await_event(self, t: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"sandbox did not emit {t!r} within {timeout}s "
                    "(bad runtime flags, missing pyodide, or a launch crash?)")
            msg = self._recv(timeout=remaining)
            if msg.get("t") == t:
                return msg

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Warm-sandbox pool — amortize the Pyodide cold start across queries
# ---------------------------------------------------------------------------
class SandboxPool:
    """A thread-safe pool of warm sandboxes.

    Starting Pyodide costs a few seconds; a pool keeps up to ``size`` subprocesses
    alive and hands them out, so that cost is paid once and amortized across many
    queries. One sandbox is leased to one caller at a time (a sandbox is a single
    stdio pipe and is not concurrency-safe); ``size`` concurrent callers get
    distinct sandboxes. Dead subprocesses are detected and replaced on acquire,
    and ``release`` clears each sandbox's state before returning it to the pool.

        pool = SandboxPool(size=4)
        rlm = RLM(client=..., sandbox_pool=pool)
        ...                      # complete() acquires/releases per call
        pool.close()

    Or directly::

        with pool.lease() as sb:
            sb.set_context(ctx); sb.run(code, handler)
    """

    def __init__(self, factory: Callable[[], Any] | None = None, size: int = 4):
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self.factory = factory or (lambda: PyodideSandbox())
        self.size = size
        self._free: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        self._count = 0          # total live sandboxes (free + leased)
        self._leased: set = set()  # checked-out sandboxes (so close() can reclaim them)
        self._closed = False

    def acquire(self, timeout: float | None = None):
        if self._closed:
            raise RuntimeError("SandboxPool is closed")
        while True:
            try:
                sb = self._free.get_nowait()
            except queue.Empty:
                with self._lock:
                    make = self._count < self.size
                    if make:
                        self._count += 1
                if make:
                    try:
                        sb = self.factory()
                        sb.start()
                    except Exception:
                        with self._lock:
                            self._count -= 1
                        raise
                    with self._lock:
                        self._leased.add(sb)
                    return sb
                try:
                    sb = self._free.get(timeout=timeout)  # block for a release
                except queue.Empty:
                    raise TimeoutError(
                        f"no free sandbox available within {timeout}s (pool size {self.size})")
            if self._alive(sb):
                with self._lock:
                    self._leased.add(sb)
                return sb
            self._discard(sb)  # dead: drop and loop to make/get another

    def release(self, sb) -> None:
        with self._lock:
            self._leased.discard(sb)
        if self._closed or not self._alive(sb):
            self._discard(sb)
            return
        try:
            sb.reset()
        except Exception:  # noqa - a sandbox that won't reset is unusable
            self._discard(sb)
            return
        self._free.put(sb)

    @contextlib.contextmanager
    def lease(self, timeout: float | None = None):
        sb = self.acquire(timeout=timeout)
        try:
            yield sb
        finally:
            self.release(sb)

    def close(self) -> None:
        self._closed = True
        leftovers = []
        while True:
            try:
                leftovers.append(self._free.get_nowait())
            except queue.Empty:
                break
        with self._lock:
            leftovers.extend(self._leased)
            self._leased.clear()
        for sb in leftovers:                 # close free AND any still-leased sandboxes
            try:
                sb.close()
            except Exception:  # noqa
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _alive(sb) -> bool:
        p = getattr(sb, "_proc", "n/a")
        if p == "n/a":
            return True  # a sandbox without a subprocess handle is treated as usable
        return p is not None and p.poll() is None

    def _discard(self, sb) -> None:
        try:
            sb.close()
        except Exception:  # noqa
            pass
        with self._lock:
            self._leased.discard(sb)
            self._count = max(0, self._count - 1)
