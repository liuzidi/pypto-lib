#!/usr/bin/env bash
# DSV4 VPTO test suite orchestrator.
# Runs setup_test_suite → setup_main → setup_vpto to generate all per-kernel dirs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Source the VPTO environment (sets PTOAS_BIN, ASCEND_HOME_PATH, etc.)
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
if [[ -f "$REPO_ROOT/scripts/vpto_env.sh" ]]; then
  source "$REPO_ROOT/scripts/vpto_env.sh" 2>/dev/null || true
fi

echo "=============================================="
echo "  DSV4 VPTO Test Suite Setup"
echo "=============================================="
echo "  ROOT_DIR         = $ROOT_DIR"
echo "  ASCEND_HOME_PATH = ${ASCEND_HOME_PATH:-unset}"
echo "  PTOAS_BIN        = ${PTOAS_BIN:-unset}"
echo ""

# Step 1: generate run dirs + golden.py stubs
echo "=== Step 1/3: setup_test_suite.py ==="
python3 setup_test_suite.py

# Step 2: generate main.cpp per kernel
echo "=== Step 2/3: setup_main.py ==="
python3 setup_main.py --mode onboard

# Step 3: generate vpto/launch.cpp + vpto/run.sh per kernel
echo "=== Step 3/3: setup_vpto.py ==="
python3 setup_vpto.py --mode onboard

echo ""
echo "Setup complete. To run all kernels:"
echo "  python3 run_all.py --device <N>"
