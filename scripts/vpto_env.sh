#!/usr/bin/env bash
# Activate the VPTO/bisheng compile environment (route 2).
#
# Route 2:  pto -> ptoas VPTO -> LLVM -> bisheng -> .o   (vs route 1 EmitC -> cpp -> g++ -> .o)
# This script only sets up the environment + rebuilds the /tmp/mlir_core_vmi
# that the ptodsl daemon needs. It does NOT wire the resulting .o into the
# golden harness (that seam is still being determined).
#
# Source it:  source scripts/vpto_env.sh
# Then run the two-step ptoas manually (see docs §11.1 / memory
# vpto-route-via-build311-ptoas).
#
# Why these paths: the VPTO backend's TileOp expansion needs the ptodsl
# daemon, which needs an mlir_core carrying the ptoas `_pto.so` binding.
# The LLVM21 build-shared mlir_core has the Python layer (ir.py etc.);
# build311's mlir_core only has the C ext. So we take LLVM21's mlir_core
# and overlay build311's _pto.so. build311 ptoas 0.53 matches this ABI
# (ptoas-bin 0.54 crashes with mixed-LLVM CommandLine errors).

set -e

# --- 1. paths (edit here if the repos move) -------------------------------
export PTOAS_SOURCE=/data/liuzidi/PTOAS
export PTOAS_BIN=/data/liuzidi/PTOAS/build311/tools/ptoas/ptoas   # 0.53, NOT ptoas-bin 0.54
export LLVM_BUILD=/data/c00862531/workspace/git/github/vpto-dev/llvm-project/build-shared
export MLIR_CORE_SRC="$LLVM_BUILD/tools/mlir/python_packages/mlir_core"
export PTO_SO=/data/liuzidi/PTOAS/build311/python/mlir/_mlir_libs/_pto.cpython-311-x86_64-linux-gnu.so
export PTO_ISA_ROOT=/data/liuzidi/pto-isa
export PTO_ISA_PATH="$PTO_ISA_ROOT"
export TILELANG_PATH="$PTOAS_SOURCE/lib/TileOps"
export TILELANG_PKG="$PTOAS_SOURCE/tilelang-dsl/python"

# --- 2. CANN — source the global set_env.sh FIRST (it may auto-detect a
# newer/different CANN install and pollute ASCEND_HOME_PATH / TOOLCHAIN_HOME /
# ASCEND_OPP_PATH), then override every CANN var back to the pinned beta.3
# toolchain that ptoas 0.53 + bisheng + the .pto files were built against. ---
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

# --- 3. rebuild /tmp/mlir_core_vmi (ptodsl daemon's mlir_core + _pto.so) ---
# /tmp is wiped on reboot; rebuild if missing or stale.
export MLIR_CORE_VMI=/tmp/mlir_core_vmi
need_rebuild=0
if [ ! -d "$MLIR_CORE_VMI/mlir" ]; then
    need_rebuild=1
elif [ "$PTO_SO" -nt "$MLIR_CORE_VMI/mlir/_mlir_libs/$(basename "$PTO_SO")" ] 2>/dev/null; then
    need_rebuild=1
fi
if [ "$need_rebuild" = "1" ]; then
    echo "[vpto_env] rebuilding $MLIR_CORE_VMI ..."
    rm -rf "$MLIR_CORE_VMI"
    cp -r "$MLIR_CORE_SRC" "$MLIR_CORE_VMI"
    cp -f "$PTO_SO" "$MLIR_CORE_VMI/mlir/_mlir_libs/"
fi

# ptodsl (source) + mlir_core_vmi (+ venv site-packages for the rest)
export PYTHONPATH="$PTOAS_SOURCE/ptodsl:$MLIR_CORE_VMI"

# --- 4. sanity ------------------------------------------------------------
echo "[vpto_env] ptoas:  $($PTOAS_BIN --version 2>&1 | head -1)"
echo "[vpto_env] vpto?  $($PTOAS_BIN --help 2>&1 | grep -E '^\s*--pto-backend=' | head -1 | sed 's/^ *//')"
python - <<'PY' 2>/dev/null && echo "[vpto_env] imports OK: mlir.ir + pto dialect + ptodsl" || echo "[vpto_env] WARN: import check failed"
from mlir.ir import Context
from mlir.dialects import pto
import ptodsl
PY
