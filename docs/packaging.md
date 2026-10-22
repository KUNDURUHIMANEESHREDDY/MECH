# Packaging the Python backend

The Windows installer ships its own CPython runtime. The app never runs on the
machine's `python`.

## Why not ship `.venv`

A virtual environment is a set of pointers. `pyvenv.cfg` records the prefix it
was created against and the `Scripts/*.exe` shims bake absolute paths, so a
copied `.venv` either fails to find its stdlib or silently resolves the system
Python. The previous build knew this and dropped the `.venv` entry at packaging
time — leaving the packaged app to start whatever `python` happened to be on
`PATH`. That is the failure mode this document replaces.

## What ships

`frontend/build/python/` is a directory that *is* the distribution:

```
python.exe, python3XX.dll, DLLs/, Lib/   CPython 3.11 base prefix
Lib/site-packages/                       backend requirements
MECH-RUNTIME.json                        what was staged, and from where
```

A copied full CPython installation resolves `Lib` and `DLLs` relative to
`python.exe`, so the directory is relocatable: no `pyvenv.cfg`, no absolute
path inside it. electron-builder copies it to `resources/python`, and
`electron/main.js:resolvePythonPath` looks there **first and only** in a
packaged build.

## Building it

```powershell
cd frontend
node scripts/python_runtime.js            # assemble from this machine (offline)
node scripts/python_runtime.js --reinstall  # pip install --target from requirements.txt
node scripts/python_runtime.js --plan     # show what would be staged
node scripts/python_runtime.js --force    # ignore the stamp and restage
```

`npm run build:electron` calls this automatically and **fails the build** if the
runtime cannot be produced. There is no warning-and-continue path: an installer
without an interpreter cannot start the backend.

Both modes require CPython 3.11 on the build machine (`MECH_PYTHON` overrides
which interpreter is probed). The staged runtime is verified by *executing* it:

```
python.exe -c "import fastapi; import uvicorn; import torch; import transformer_lens; import numpy; import pandas"
```

If that fails, staging fails and the build stops. The requirements are the
repository's `requirements.txt`, whose upper bounds (`numpy<2`, `pandas<2.1`)
are load-bearing for the transformer-lens stack.

## Strategy tradeoffs

| Mode | Network | Reproducibility | Notes |
| ---- | ------- | --------------- | ----- |
| `installed` (default) | none | interpreter version + `requirements.txt` hash recorded in `MECH-RUNTIME.json`; site-packages come from the build machine's environment | fastest; the build machine must already have the requirements installed |
| `reinstall` | required | `pip install --target` resolves `requirements.txt` from scratch into the staged tree | the release-grade path; use it for anything published |
| `reinstall --cpu` | required | as above, then torch is force-reinstalled from the CPU index and `nvidia*`/`cudnn*`/`nccl*`/`triton*` are pruned | the release profile for inference-only installs; see *Size* |

## Size

The default `requirements.txt` resolves torch to the CUDA wheel, which pulls
several GiB of `nvidia*` runtime libraries. A runtime staged that way is about
**5.6 GiB / 50k files**, and the installer is correspondingly large.

For a release that only needs inference:

```powershell
node scripts/python_runtime.js --reinstall --cpu
```

This installs `torch` from `https://download.pytorch.org/whl/cpu` and removes the
CUDA runtime packages, which is what dominates the size. `--cpu` implies
`--reinstall`: an already-installed CUDA build cannot be converted offline, and
staging refuses rather than shipping a CUDA wheel while claiming to be CPU-only.

The staged size is printed at the end of staging and recorded as `sizeBytes` in
`MECH-RUNTIME.json`, so a size regression is visible in the build log.

`MECH-RUNTIME.json` records the interpreter version, the requirements hash, the
staging mode and the verified imports. A restage is skipped only when all three
match, so a changed `requirements.txt` always rebuilds.

## Packaging

`frontend/package.json` declares exactly two required `extraResources`:

* `build/backend` → `resources/backend`
* `build/python` → `resources/python`

`scripts/build.js` stages both before electron-builder runs and exits non-zero
if either is missing.

## Verifying on a clean machine

This is the only test that proves portability, and it cannot be done from the
build machine:

1. Copy `frontend/release/MECH Platform Setup 2.0.0.exe` to a Windows machine
   with **no** Python installed (or with an unrelated Python on `PATH`).
2. Install and launch.
3. Expect: the window opens with the backend connected. The log
   (`%APPDATA%/MECH/logs`) contains `backend_spawning` with a path under
   `...\resources\python\python.exe`.
4. Negative check: rename `resources\python` and relaunch. Expect a dialog
   "MECH backend failed to start" naming the expected interpreter path — never a
   silent fallback to the system Python.

Record the result in the release notes. Until step 3 has been performed on a
clean machine, the installer is **unverified**, not "working".
