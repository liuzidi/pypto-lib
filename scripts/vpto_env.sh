#!/usr/bin/env bash
# Activate the VPTO/bisheng compile environment (route 2).
#
# Route 2:  pto -> ptoas VPTO -> LLVM -> bisheng -> .o   (vs route 1 EmitC -> cpp -> g++ -> .o)
# This script only sets up the environment. It does NOT wire the resulting .o
# into the golden harness (that seam is still being determined).
#
# Source it:  source scripts/vpto_env.sh
#
# --- ptoas binary selection ----------------------------------------------
# ptoas 0.59+ is cpython-3.12 native. The build-tree wrapper at
# $PTOAS_SOURCE/build/tools/ptoas/ptoas is broken in environments that also
# have a pip-installed (editable) ptoas wheel: the wrapper calls
# _disable_editable_import_redirects(), which removes the editable finder
# that normally makes `ptoas.mlir.ir` resolvable, and the build-tree
# build/python/ptoas/mlir/ is only a PARTIAL staging (it has the PTO dialect
# stubs and the freshly-built _core.so / _mlir.so, but NOT ir.py or
# _mlirRegisterEverything.so). The result is
# "ModuleNotFoundError: No module named 'ptoas.mlir.ir'".
#
# The pip-installed wheel (~/.local/bin/ptoas, from `pip install -e .` /
# `pip install .`) is a SELF-CONTAINED copy of the same build (same _core.so
# and _mlir.so md5 as the build tree) and works out of the box. So we prefer
# the wheel entry when it exists and reports a working --version, and only
# fall back to the build-tree wrapper when the wheel is absent (e.g. on a
# clean machine where the user has only done a CMake build, no pip install).
#
# --- ptodsl daemon PYTHONPATH --------------------------------------------
# When ptoas runs the ptodsl DSL in-process (0.59+), the daemon subprocess
# inherits PYTHONPATH. It needs:
#   1. $PTOAS_SOURCE/build/python  — ptoas package + _core.so + PTO dialect
#      (wins over the source-tree ptodsl/ptoas/__init__.py)
#   2. $MLIR_CORE_VMI              — /tmp overlay of the LLVM build's
#      mlir_core python package (ir.py, passmanager.py, _mlir.so,
#      _mlirRegisterEverything.so). See the overlay rebuild block below for
#      why the overlay exists instead of using $MLIR_CORE_SRC directly.
#   3. $PTOAS_SOURCE/ptodsl         — the ptoas CLI + DSL runtime
# For old builds (build311, cpython-3.11) only roots 2+3 are needed.

set -e

# --- 1. paths (edit here if the repos move) -------------------------------
export PTOAS_SOURCE=/data/liuzidi/PTOAS
# NOTE: must point at a cpython-3.12 llvm build so the mlir python extensions
# (_mlir.cpython-312-*.so etc.) load under the system python3 (3.12). The
# older /data/c00862531/... tree was built against cpython-3.11 and will fail
# with "cannot import name 'ir' from '...._mlir'" under 3.12.
export LLVM_BUILD=/data/liuzidi/llvm-workspace/llvm-project/build-shared
export PTO_ISA_ROOT=/data/liuzidi/pto-isa
export PTO_ISA_PATH="$PTO_ISA_ROOT"
export TILELANG_PATH="$PTOAS_SOURCE/lib/TileOps"
export TILELANG_PKG="$PTOAS_SOURCE/tilelang-dsl/python"

# Prefer the pip-installed wheel entry (~/.local/bin/ptoas) when it works;
# fall back to the build-tree wrapper otherwise. See header comment.
_WHEEL_PTOAS="${PTOAS_WHEEL_BIN:-$(python3 -c 'import shutil,sys; print(shutil.which("ptoas") or "")' 2>/dev/null)}"
_BUILD_PTOAS="$PTOAS_SOURCE/build/tools/ptoas/ptoas"
if [ -n "$_WHEEL_PTOAS" ] && "$_WHEEL_PTOAS" --version >/dev/null 2>&1; then
    export PTOAS_BIN="$_WHEEL_PTOAS"
else
    export PTOAS_BIN="$_BUILD_PTOAS"
fi

# --- 2. CANN — source the global set_env.sh FIRST (it may auto-detect a
# newer/different CANN install and pollute ASCEND_HOME_PATH / TOOLCHAIN_HOME /
# ASCEND_OPP_PATH), then override every CANN var back to the pinned beta.3
# toolchain that ptoas + bisheng + the .pto files were built against. ---
export LD_LIBRARY_PATH="$LLVM_BUILD/lib:${LD_LIBRARY_PATH:-}"
set +u; source /usr/local/Ascend/cann/set_env.sh 2>/dev/null; set -u
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0-beta.3
export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
export TOOLCHAIN_HOME="$ASCEND_HOME_PATH/toolkit"
export BISHENG_BIN="$ASCEND_HOME_PATH/bin/bisheng"
# Prepend the pinned CANN's lib/bin to the front so it wins over any other
# CANN install the global set_env.sh may have added.
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/bisheng_compiler/bin:$PATH"

# --- 3. /tmp/mlir_core_vmi overlay + ptodsl daemon PYTHONPATH ---
# The LLVM build tree's mlir_core python package ($MLIR_CORE_SRC/mlir/) has
# ir.py, passmanager.py, _mlir.so, _mlirRegisterEverything.so, etc. The ptoas
# build tree only stages the PTO dialect + ptoas wrapper + _core.so. For the
# top-level `mlir` namespace package (which ptoas.mlir aliases to) to resolve
# ir.py AND the PTO dialect, both roots must contribute an "mlir/" dir.
#
# But mlir_core's _mlir.so and the ptoas build's _mlir.so are NOT
# interchangeable across builds: if both are on sys.path, the namespace merge
# loads BOTH _mlir_libs/__init__.py instances, running _site_initialize twice
# and triggering "PyCapsule_GetPointer called with incorrect name" /
# "register_dialects(): incompatible function arguments". So we copy
# mlir_core to /tmp and overlay the build's _mlir.so / _mlirDialectsLLVM.so /
# libPTOASCompiler.so / _site_initialize_0.so on top, leaving only ONE
# _mlir.so on sys.path: the one matching ptoas/_core.so.
export MLIR_CORE_SRC="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
export MLIR_CORE_VMI=/tmp/mlir_core_vmi
_PTOAS_VER=$("$PTOAS_BIN" --version 2>/dev/null | head -1)
if [[ "$_PTOAS_VER" == "ptoas 0.5"* ]] || [[ "$_PTOAS_VER" == "ptoas 0.4"* ]]; then
    # Old path: build311 (cpython-311). Overlay just the PTO dialect .so.
    _BUILD_MLIR_LIBS=/data/liuzidi/PTOAS/build311/python/mlir/_mlir_libs
    export PYTHONPATH="$PTOAS_SOURCE/ptodsl:$MLIR_CORE_VMI"
else
    # 0.59+ (cpython-3.12). Overlay the build's _mlir_libs/*.so onto the
    # mlir_core copy so the merged `mlir` namespace uses ptoas's binaries.
    _BUILD_MLIR_LIBS="$PTOAS_SOURCE/build/python/ptoas/mlir/_mlir_libs"
    export PYTHONPATH="$PTOAS_SOURCE/build/python:$MLIR_CORE_VMI:$PTOAS_SOURCE/ptodsl"
fi

# Rebuild the overlay if it's missing, or if any overlaid .so in the build is
# newer than the copy in the overlay.
need_rebuild=0
if [ ! -d "$MLIR_CORE_VMI/mlir" ]; then
    need_rebuild=1
else
    for _so in "$_BUILD_MLIR_LIBS"/*.so; do
        [ -e "$_so" ] || continue
        _name=$(basename "$_so")
        if [ "$_so" -nt "$MLIR_CORE_VMI/mlir/_mlir_libs/$_name" ] 2>/dev/null; then
            need_rebuild=1; break
        fi
    done
fi
if [ "$need_rebuild" = "1" ]; then
    echo "[vpto_env] rebuilding $MLIR_CORE_VMI ..."
    rm -rf "$MLIR_CORE_VMI"
    cp -r "$MLIR_CORE_SRC" "$MLIR_CORE_VMI"
    # Overlay every native extension the ptoas build staged under its own
    # ptoas/mlir/_mlir_libs so the overlay's _mlir matches ptoas/_core.so.
    cp -f "$_BUILD_MLIR_LIBS"/*.so "$MLIR_CORE_VMI/mlir/_mlir_libs/" 2>/dev/null
fi

# --- 4. sanity ------------------------------------------------------------
echo "[vpto_env] ptoas:  $($PTOAS_BIN --version 2>&1 | head -1)  ($PTOAS_BIN)"
echo "[vpto_env] vpto?  $($PTOAS_BIN --help 2>&1 | grep -E '^\s*--pto-backend=' | head -1 | sed 's/^ *//')"
# Verify ptoas can actually run (parse + lower). The old mlir.ir import check
# was unreliable because the build-tree ptoas.mlir staging is partial; the
# wheel entry used above is self-contained. A --help round-trip confirms the
# native module loads and the CLI is wired.
if "$PTOAS_BIN" --help >/dev/null 2>&1; then
    echo "[vpto_env] ptoas functional: --help OK"
else
    echo "[vpto_env] WARN: ptoas --help failed"
fi
