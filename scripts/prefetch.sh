#!/usr/bin/env bash
# Pre-fetch the pinned Pyodide so the sandbox runs with no network at runtime.
# Deno is the hardened default; Node is the dev fallback.
set -euo pipefail
PYODIDE_VERSION="${PYODIDE_VERSION:-314.0.0}"

if command -v deno >/dev/null 2>&1; then
  echo "[prefetch] caching npm:pyodide@${PYODIDE_VERSION} for Deno..."
  deno cache "npm:pyodide@${PYODIDE_VERSION}"
  echo "[prefetch] DENO_DIR = $(deno info --json | python3 -c 'import sys,json;print(json.load(sys.stdin)["denoDir"])')"
  echo "[prefetch] done. Runtime needs no network; set DENO_DIR to this path if it is non-default."
elif command -v node >/dev/null 2>&1; then
  echo "[prefetch] installing pyodide@${PYODIDE_VERSION} for Node (dev fallback)..."
  npm install "pyodide@${PYODIDE_VERSION}"
  echo "[prefetch] done."
else
  echo "[prefetch] ERROR: neither deno nor node found on PATH. Install Deno (recommended)." >&2
  exit 1
fi
