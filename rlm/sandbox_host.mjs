// sandbox_host.mjs — runs inside Deno (preferred) or Node.
//
// Hosts a Pyodide (CPython-in-WASM) interpreter and speaks newline-delimited
// JSON over stdio with the Python host process (rlm/sandbox.py):
//
//   host -> sandbox : {t:"init", context, contextKind, head, tail}
//                     {t:"run", id, code}
//   sandbox -> host : {t:"ready"}
//                     {t:"sub", id, runId, kind:"query"|"batch", payload}
//                     {t:"result", id, stdout, final, hasFinal, finalIsJson}
//                     {t:"inited"}
//   host -> sandbox : {t:"sub_result", id, value}
//
// On {t:"init"} the Python global namespace is reset (leaked user variables from
// a prior query are cleared) before the new `context` is bound — so a reused
// sandbox does not bleed state across queries.
//
// Security model: the generated Python runs in the WASM boundary with NO network
// and NO API keys. The only way out is the `sub` channel, which the host fully
// controls. Under Deno the process additionally runs with denied permissions.

const IS_DENO = typeof Deno !== "undefined";

const enc = new TextEncoder();
function writeLine(obj) {
  const line = JSON.stringify(obj) + "\n";
  if (IS_DENO) Deno.stdout.writeSync(enc.encode(line));
  else process.stdout.write(line);
}

async function* stdinLines() {
  let buf = "";
  if (IS_DENO) {
    const dec = new TextDecoder();
    const chunk = new Uint8Array(65536);
    while (true) {
      const n = await Deno.stdin.read(chunk);
      if (n === null) break;
      buf += dec.decode(chunk.subarray(0, n), { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (line.trim()) yield line;
      }
    }
  } else {
    process.stdin.setEncoding("utf8");
    const queue = [];
    let resolve = null;
    process.stdin.on("data", (d) => {
      buf += d;
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 1);
        if (line.trim()) { if (resolve) { resolve(line); resolve = null; } else queue.push(line); }
      }
    });
    while (true) {
      if (queue.length) { yield queue.shift(); continue; }
      yield await new Promise((r) => (resolve = r));
    }
  }
}

// ---- load Pyodide -----------------------------------------------------------
let loadPyodide;
if (IS_DENO) ({ loadPyodide } = await import("npm:pyodide"));
else ({ loadPyodide } = await import("pyodide"));

const indexURL = (() => {
  const fromArg = (IS_DENO ? Deno.args[0] : process.argv[2]);
  return fromArg || undefined;
})();

const py = await loadPyodide(indexURL ? { indexURL } : {});

const pending = new Map();
let subSeq = 0;
let currentRunId = null;
let ctxKind = null;
let ctxParts = [];

function makeSub(kind) {
  return (payload) => {
    const id = "s" + (++subSeq);
    const p = new Promise((resolve) => pending.set(id, resolve));
    const plain = payload && payload.toJs ? payload.toJs({ dict_converter: Object.fromEntries }) : payload;
    writeLine({ t: "sub", id, runId: currentRunId, kind, payload: plain });
    return p;
  };
}
py.globals.set("__host_query", makeSub("query"));
py.globals.set("__host_batch", makeSub("batch"));
py.globals.set("__host_rlm", makeSub("rlm_query"));
py.globals.set("__host_rlm_batch", makeSub("rlm_batch"));
py.globals.set("__host_tool", makeSub("tool"));

// ---- Python-side bootstrap: isolation guards + API + reset ------------------
const BOOTSTRAP = `
import sys, io, json, asyncio

# Defense-in-depth: remove Pyodide's JsFinder so 'import js' cannot bridge to the
# host JS runtime. (The hard boundary is the runtime's denied permissions under Deno.)
sys.meta_path = [m for m in sys.meta_path if 'pyodide' not in type(getattr(m,'__class__',m)).__module__]
for _bad in ('js', 'pyodide_js', 'pyodide', 'ctypes', 'socket'):
    sys.modules[_bad] = None

class _Final(Exception):
    def __init__(self, value=None): self.value = value

def FINAL(value=None):
    raise _Final(value)

def FINAL_VAR(name):
    raise _Final(globals().get(str(name).strip().strip("'\\"")))

async def llm_query(task, data=None, schema=None):
    req = {"task": task, "data": data, "schema": schema}
    return await __host_query(req)

async def batch_llm_query(jobs):
    jobs = [dict(j) for j in jobs]
    return list(await __host_batch(jobs))

async def rlm_query(task, data=None, schema=None):
    # Like llm_query, but spawns a full recursive child RLM (its own REPL) over the
    # data. Falls back to a plain leaf call at the recursion-depth budget.
    return await __host_rlm({"task": task, "data": data, "schema": schema})

async def batch_rlm_query(jobs):
    jobs = [dict(j) for j in jobs]
    return list(await __host_rlm_batch(jobs))

async def tool(__tool_name, /, *args, **kwargs):
    # Call a host-registered tool. Result is whatever the host returns (JSON).
    # __tool_name is positional-only so a tool may itself take a name= kwarg.
    return await __host_tool({"name": __tool_name, "args": list(args), "kwargs": dict(kwargs)})

def __rlm_reset__():
    g = globals()
    for _k in [k for k in g if k not in __RLM_KEEP__]:
        del g[_k]

# Captured AFTER __rlm_reset__ is defined so the function (and this set) survive a
# reset. Everything a model defines during a query is cleared on the next init.
__RLM_KEEP__ = set(globals().keys()) | {'__RLM_KEEP__'}
`;
await py.runPythonAsync(BOOTSTRAP);

function buildRunner(head, tail) {
  return (codeJson) => `
import io, sys, json, ast
_code = json.loads(r'''${codeJson}''')
_buf = io.StringIO(); _old = sys.stdout; sys.stdout = _buf
_res = {"stdout": "", "final": None, "hasFinal": False, "finalIsJson": True}
try:
    _tree = ast.parse(_code)
    _echo = bool(_tree.body) and isinstance(_tree.body[-1], ast.Expr)
    if _echo:
        _last = _tree.body[-1]
        _assign = ast.Assign(
            targets=[ast.Name(id='__cell_result__', ctx=ast.Store())],
            value=_last.value)
        ast.copy_location(_assign, _last)
        _tree.body[-1] = _assign
        ast.fix_missing_locations(_tree)
    _compiled = compile(_tree, '<cell>', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    globals().pop('__cell_result__', None)
    _maybe = eval(_compiled, globals())
    if _maybe is not None:
        await _maybe
    if _echo and globals().get('__cell_result__') is not None:
        print(repr(globals()['__cell_result__']))
except _Final as _f:
    _res["hasFinal"] = True
    try:
        json.dumps(_f.value); _res["final"] = _f.value
    except (TypeError, ValueError):
        _res["final"] = repr(_f.value); _res["finalIsJson"] = False
except Exception as _e:
    print(repr(_e))
finally:
    sys.stdout = _old
    globals().pop('__cell_result__', None)
_out = _buf.getvalue()
_HEAD, _TAIL = ${head}, ${tail}
if len(_out) > _HEAD + _TAIL:
    _omitted = len(_out) - _HEAD - _TAIL
    _out = _out[:_HEAD] + f"\\n... [{_omitted} chars truncated by scaffold] ...\\n" + _out[-_TAIL:]
_res["stdout"] = _out or "(no output)"
json.dumps(_res)
`;
}

async function bindContext(kind, payload) {
  if (kind === "json") {
    py.globals.set("context", py.toPy(JSON.parse(payload)));
  } else if (kind === "pickle") {
    py.globals.set("__ctx_b64__", payload);
    await py.runPythonAsync(
      "import base64, pickle\ntry:\n    context = pickle.loads(base64.b64decode(__ctx_b64__))\nfinally:\n    del __ctx_b64__");
  } else {
    py.globals.set("context", payload);  // str
  }
}

let runner = buildRunner(4000, 1000);

writeLine({ t: "ready" });
for await (const line of stdinLines()) {
  let msg;
  try { msg = JSON.parse(line); } catch { continue; }

  if (msg.t === "init") {
    runner = buildRunner(msg.head ?? 4000, msg.tail ?? 1000);
    await py.runPythonAsync("__rlm_reset__()");   // clear leaked user globals
    try { await bindContext(msg.contextKind, msg.payload); writeLine({ t: "inited" }); }
    catch (e) { writeLine({ t: "inited", error: String(e) }); }
  } else if (msg.t === "init_begin") {
    runner = buildRunner(msg.head ?? 4000, msg.tail ?? 1000);
    await py.runPythonAsync("__rlm_reset__()");
    ctxKind = msg.contextKind;
    ctxParts = [];
  } else if (msg.t === "ctx_chunk") {
    ctxParts.push(msg.data);
  } else if (msg.t === "init_end") {
    const assembled = ctxParts.join("");
    ctxParts = [];
    try { await bindContext(ctxKind, assembled); writeLine({ t: "inited" }); }
    catch (e) { writeLine({ t: "inited", error: String(e) }); }
  } else if (msg.t === "reset") {
    await py.runPythonAsync("__rlm_reset__()");
    writeLine({ t: "reset_done" });
  } else if (msg.t === "sub_result") {
    const resolve = pending.get(msg.id);
    if (resolve) { pending.delete(msg.id); resolve(msg.value); }
  } else if (msg.t === "run") {
    currentRunId = msg.id;
    py.runPythonAsync(runner(JSON.stringify(msg.code)))
      .then((out) => writeLine({ t: "result", id: msg.id, ...JSON.parse(out) }))
      .catch((e) => writeLine({
        t: "result", id: msg.id, stdout: "SandboxError: " + String(e),
        final: null, hasFinal: false, finalIsJson: false,
      }));
  } else if (msg.t === "close") {
    break;
  }
}
if (!IS_DENO) process.exit(0);
