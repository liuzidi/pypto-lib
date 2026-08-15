#!/usr/bin/env python3
"""Generate run_<kernel>/ dirs + golden.py stubs for each DSV4 kernel.

Mirrors test_for_ptoas/setup_test_suite.py but for DSV4 kernels.
Reads BUILDERS from dsv4_golden_lib.py, intersects with .pto files,
creates per-kernel run dirs with golden.py stub + outputs.txt.
"""
import ast
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PTO_DIR = ROOT / ".pto"
GOLDEN_LIB = ROOT / "dsv4_golden_lib.py"

GOLDEN_PY_TEMPLATE = '''#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv4_golden_lib import run_case
if __name__ == "__main__":
    run_case("{kernel}")
'''

def extract_builder_names(golden_lib: Path) -> list:
    """AST-parse the BUILDERS dict to get registered kernel names."""
    tree = ast.parse(golden_lib.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BUILDERS":
                    if isinstance(node.value, ast.Dict):
                        return [k.value for k in node.value.keys
                                if isinstance(k, ast.Constant)]
                    elif isinstance(node.value, ast.Call):
                        # BUILDERS = {name: fn for ...} or BUILDERS = dict(...)
                        return []
    return []

def main():
    if not GOLDEN_LIB.exists():
        print(f"ERROR: {GOLDEN_LIB} not found. Write it first.", file=sys.stderr)
        return 1

    # Get builder names
    builders = extract_builder_names(GOLDEN_LIB)
    print(f"BUILDERS has {len(builders)} entries: {builders[:5]}...")

    # Get .pto files
    pto_files = {p.stem: p for p in sorted(PTO_DIR.glob("*.pto"))}
    print(f"Found {len(pto_files)} .pto files")

    # Intersect
    kernels = sorted(set(builders) & set(pto_files.keys()))
    missing_pto = set(builders) - set(pto_files.keys())
    missing_builder = set(pto_files.keys()) - set(builders)
    print(f"Matched {len(kernels)} kernels")
    if missing_pto:
        print(f"  WARN: {len(missing_pto)} builders without .pto: {sorted(missing_pto)[:5]}")
    if missing_builder:
        print(f"  WARN: {len(missing_builder)} .pto without builders: {sorted(missing_builder)[:5]}")

    # Generate run dirs
    for kname in kernels:
        run_dir = ROOT / f"run_{kname}"
        run_dir.mkdir(exist_ok=True)

        # golden.py stub
        golden_py = run_dir / "golden.py"
        golden_py.write_text(GOLDEN_PY_TEMPLATE.format(kernel=kname))

        # outputs.txt (empty — will be filled by setup_main.py from golden_lib)
        out_txt = run_dir / "outputs.txt"
        if not out_txt.exists():
            out_txt.write_text("")

        # compare.py symlink
        cmp = run_dir / "compare.py"
        if not cmp.exists() and not cmp.is_symlink():
            cmp.symlink_to("../compare.py")

    print(f"\nGenerated {len(kernels)} run_<kernel>/ dirs")
    print("Next: run setup_main.py --mode onboard && setup_vpto.py --mode onboard")
    return 0

if __name__ == "__main__":
    sys.exit(main())
