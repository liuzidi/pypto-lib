#!/usr/bin/env python3
"""DSV4 VPTO golden reference library.

Self-contained CPU golden for all 110 DSV4 kernels. No GM dump, no NPU capture.
Each build_<name>(meta, generator, ints) returns (buffers_dict, golden_dict).
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "golden_parts"))

# Import validation_runtime so part files can `from validation_runtime import ...`
from validation_runtime import (
    load_case_meta, rng, load_int32_assignments,
    write_buffers, write_golden,
    bf16_to_float32, float32_to_bf16,
)

# Exec each part file in its own namespace, merge BUILDERS
_ns = {}
BUILDERS = {}
for _part in sorted((_HERE / "golden_parts").glob("*.py")):
    _pns = {}
    exec(compile(_part.read_text(), str(_part), "exec"), _pns)
    if "BUILDERS" in _pns:
        BUILDERS.update(_pns["BUILDERS"])
    _ns.update({k: v for k, v in _pns.items() if not k.startswith("__")})

# Export all public names (functions, constants) into module globals
for _name, _val in _ns.items():
    if not _name.startswith("_") and _name not in globals():
        globals()[_name] = _val

# Aliases for missing kernels
if "build_kv_rms_norm_rope" in _ns:
    BUILDERS.setdefault("rmsnorm_rope", _ns["build_kv_rms_norm_rope"])
    # rmsnorm_rope_cache_write has its own dedicated builder
    # (build_rmsnorm_rope_cache_write) registered via the part-file BUILDERS.




# ===== Output buffer names for get_outputs_from_golden_lib regex =====
# These fake functions exist ONLY so get_outputs_from_golden_lib's regex
# can extract output buffer names. run_case uses BUILDERS dict, not these.
def build_build_bias():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_comb_sinkhorn():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_gather_kv():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_idx_qr_proj_dequant():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_idx_qr_proj_matmul():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_kv_and_cache_write():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0), "v3": np.zeros(0), "v3": np.zeros(0), "v6": np.zeros(0), "v6": np.zeros(0)}

def build_kv_hadamard():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_kv_proj_matmul():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_kv_proj_seed():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_kv_rms_norm_rope():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_kv_score_proj():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_kv_score_proj_0():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_kv_touch():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_merge_norm():
    buffers = {}
    return buffers, {"v8": np.zeros(0), "v8": np.zeros(0)}

def build_mix_x():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_mtp_projection_linear():
    buffers = {}
    return buffers, {"v7": np.zeros(0), "v7": np.zeros(0)}

def build_mtp_projection_norm():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0)}

def build_mtp_projection_output():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_mtp_projection_quant():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0), "v3": np.zeros(0), "v3": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0), "v6": np.zeros(0), "v6": np.zeros(0)}

def build_mtp_projection_rms():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0), "v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_c4_cache_write():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_c4_kv_score_proj():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v5": np.zeros(0)}

def build_prefill_c4_rmsnorm_rope():
    buffers = {}
    return buffers, {"v6": np.zeros(0), "v6": np.zeros(0)}

def build_prefill_c4_softmax_pool():
    buffers = {}
    return buffers, {"v9": np.zeros(0), "v9": np.zeros(0)}

def build_prefill_c4_state_update():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_c4_write_map():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_csa_cache_write():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_csa_idx_halfrope():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_hca_c128_kv_finalize():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_hca_c128_kv_score_proj():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v5": np.zeros(0)}

def build_prefill_hca_c128_norm_pad_init():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_hca_c128_rmsnorm_rope():
    buffers = {}
    return buffers, {"v6": np.zeros(0), "v6": np.zeros(0)}

def build_prefill_hca_c128_softmax_pool():
    buffers = {}
    return buffers, {"v5": np.zeros(0), "v5": np.zeros(0)}

def build_prefill_hca_c128_state_scatter_pre():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_hca_c128_write_map():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_hca_cache_write():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_idx_c4_cache_write():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_idx_c4_kv_hadamard():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_idx_c4_kv_score_proj():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v5": np.zeros(0)}

def build_prefill_idx_c4_rmsnorm_rope():
    buffers = {}
    return buffers, {"v7": np.zeros(0), "v7": np.zeros(0)}

def build_prefill_idx_c4_softmax_pool():
    buffers = {}
    return buffers, {"v9": np.zeros(0), "v9": np.zeros(0)}

def build_prefill_idx_c4_state_update():
    buffers = {}
    return buffers, {"v6": np.zeros(0), "v6": np.zeros(0)}

def build_prefill_idx_c4_write_map():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0)}

def build_prefill_idx_qr_hadamard_quant():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_idx_qr_proj():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_idx_qr_rope():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_prefill_idx_score():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_idx_score_init():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_idx_score_out():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_prefill_idx_topk():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_prefill_idx_weights_proj():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_proj_a_mm():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_proj_b_act():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_proj_b_mm():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_q_rope_prepare():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_qk_pv():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_qkv_rope_rows():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_qproj_matmul():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_qr_hadamard_matmul():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_qr_hadamard_quant():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0), "v3": np.zeros(0), "v3": np.zeros(0)}

def build_qr_proj_matmul():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_qr_proj_seed():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_qr_rms_norm_quant():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0), "v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_qr_rope():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0)}

def build_quant():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_rms_norm():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_rmsnorm_rope():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}

def build_rmsnorm_rope_cache_write():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v6": np.zeros(0), "v7": np.zeros(0)}

def build_rope():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_rope_cs():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0), "v3": np.zeros(0), "v3": np.zeros(0)}

def build_route_hash():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_scatter_softmax_pool():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_score_mat():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_score_reduce():
    buffers = {}
    return buffers, {"v5": np.zeros(0), "v5": np.zeros(0)}

def build_split_pre_post():
    buffers = {}
    return buffers, {"v4": np.zeros(0), "v4": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_swa_cache_insert_valid_bias():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v5": np.zeros(0), "v5": np.zeros(0)}

def build_swa_gather_kv():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0)}

def build_swa_rope_step():
    buffers = {}
    return buffers, {"v1": np.zeros(0), "v1": np.zeros(0), "v2": np.zeros(0), "v2": np.zeros(0)}

def build_topk():
    buffers = {}
    return buffers, {"v1": np.zeros(0)}

def build_weights_proj():
    buffers = {}
    return buffers, {"v3": np.zeros(0), "v3": np.zeros(0)}

def build_weights_proj_reduce():
    buffers = {}
    return buffers, {"v2": np.zeros(0), "v2": np.zeros(0)}


def run_case(name):
    """Generate golden inputs + expected output for kernel `name`.

    Called by run_<kernel>/golden.py. Reads main.cpp for buffer metadata,
    generates random inputs with seeded RNG, computes golden on CPU.
    """
    meta = load_case_meta()
    generator = rng()
    ints = load_int32_assignments()
    # Split kernels (gate_aic, gate_aiv) share a base builder named after the
    # base kernel (gate). The _aic variant is the cube matmul; the _aiv variant
    # is the vector epilogue. Both share the same input/output buffers, so the
    # base builder produces the correct golden for the _aic part. The _aiv
    # part may produce slightly different intermediate results but uses the
    # same output buffer shape.
    lookup = name
    if lookup not in BUILDERS:
        import re
        base = re.sub(r'_(aic|aiv)$', '', lookup)
        if base in BUILDERS:
            lookup = base
        else:
            raise KeyError(f"kernel {name!r} (tried {base!r}) not in BUILDERS "
                          f"(available: {sorted(BUILDERS)[:5]}...)")
    fn = BUILDERS[lookup]
    buffers, golden = fn(meta, generator, ints)
    write_buffers(meta, buffers)
    write_golden(meta, golden)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel_name>", file=sys.stderr)
        sys.exit(1)
    run_case(sys.argv[1])
