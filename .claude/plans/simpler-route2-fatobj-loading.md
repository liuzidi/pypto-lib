# Simpler Route-2 Fatobj Loading — Design for Review

> Proposal: a small, localized change to pypto's `device_runner.py` +
> `elf_parser.py` so that simpler can load VPTO Route-2 bisheng fat-objects
> the same way it loads Route-1 plain `.o` files. This would let DSV4
> modules run through `golden.run_jit(...)` end-to-end on Route 2 — the
> same one-command experience Qwen has on Route 1 — instead of the current
> per-kernel `vpto_run.py` harness.

## 1. Background — two routes, one runtime

DSV4 kernels go from Python DSL to A5 NPU through this pipeline:

```
@pl.jit (pypto DSL)
  → .pto (PTO MLIR IR)          [pypto frontend tracer]
  → fatobj .o                   [ptoas VPTO backend → LLVM → bisheng fatobj]
  → lib<kernel>_kernel.so       [bisheng --cce-fatobj-link]
  → NPU execute + compare       [CANN module-load]
```

**Route 1 (EmitC)** and **Route 2 (VPTO)** differ only at the ptoas step:
which backend lowers `.pto` → `.o`. Everything downstream (golden, compare,
the CANN launch API) is shared in principle.

### The current state of each route

| Route | ptoas backend | `.o` layout | simpler loads it? | User experience |
|-------|---------------|-------------|-------------------|------------------|
| 1 (EmitC) | `--pto-backend=emitc` | plain `.o`, `.text` = raw A5 inst stream | ✅ yes (all 27 DSV4 baseline ops PASS) | `python <model>.py -p a5 -d 0` (run_jit, module-level) |
| 2 (VPTO) | `--pto-backend=vpto` | bisheng fatobj, device code in `__aicore_rel_binary` nested ELF, outer `.text` = 197-B host glue (not device code) | ❌ no (extract_text_section reads the host glue) | per-kernel `vpto_run.py --pto ... --model-py ...` |

Route 1 gives the desired Qwen-style one-command, module-level experience
(non-leaf kernels run automatically because simpler drives the whole module).
Route 2 currently requires a **per-kernel** harness (`vpto-board-validate`
skill) because simpler can't load the fatobj — so non-leaf kernels can't run
through the golden harness at all (Phase 5 intermediate capture is the
**planned but unimplemented** fix, per `vpto-dsv4-vector-validation.md:107`
"future, out of scope now"; today non-leaf modules simply fail at the skill's
"not a leaf module" / "could not map ptr" check).

**The goal of this change:** make simpler load Route-2 fatobjs so Route 2
also gets the module-level `run_jit` experience.

## 2. Root cause — it's a Python bug, not a missing C++ loader

Earlier notes (now superseded) claimed simpler "has no CANN module-load
path" and loads "raw A5 instruction bytes" via a raw-bytes InCore model.
**That is wrong.** Verified against the current simpler pin
(`runtime` revision `3165cc89`, 2026-08-04):

### 2.1 simpler already calls the CANN module-load API

`DeviceRunnerBase::launch_aicore_kernel` at
`runtime/src/common/platform/onboard/host/device_runner_base.cpp:1229`:

```cpp
int DeviceRunnerBase::launch_aicore_kernel(rtStream_t stream, KernelArgs *k_args) {
    if (aicore_bin_handle_ == nullptr) {
        rtDevBinary_t binary;
        std::memset(&binary, 0, sizeof(binary));
        binary.magic = RT_DEV_BINARY_MAGIC_ELF;   // 0x43554245 = ASCII "CUBE"
        binary.version = 0;
        binary.data = aicore_kernel_binary_.data();
        binary.length = aicore_kernel_binary_.size();
        int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);  // CANN module load
        ...
    }
    int rc = rtKernelLaunchWithHandleV2(aicore_bin_handle_, ..., stream, &cfg);
    ...
}
```

Key facts (verified against
`/usr/local/Ascend/cann-9.1.0-beta.3/x86_64-linux/pkg_inc/runtime/rt_external_kernel.h`):

- `RT_DEV_BINARY_MAGIC_ELF = 0x43554245` is **not** an ELF-magic check — it is
  a CANN **kernel-type tag** (ASCII `"CUBE"`; siblings `_AICPU`/`_AIVEC`/`_AICUBE`
  exist). `rtRegisterAllKernel` accepts a tagged byte blob.
- `aicore_kernel_binary_` is a `std::vector<uint8_t>` filled once at init by
  `set_executors(aicpu_so, aicore_kernel)`.

**So simpler already has the CANN module-load path.** The C++ side needs
**no change**. The "raw-bytes InCore" framing in old notes was describing the
*contents* of the blob, not a separate loading mechanism.

### 2.2 The real seam: `.text`-section stripping in the Python layer

The producer is `pypto/python/pypto/runtime/device_runner.py:514` inside
`compile_single_kernel` (def @457; note `compile_and_assemble` @700 merely
delegates to it — verified 2026-08-12):

```python
kernel_bin = raw if platform.endswith("sim") else extract_text_section(raw)
#                ^^^ sim: feed whole .so        ^^^ HW: feed ONLY .text
```

- **Route 1 `.o`** (EmitC→ccec incore toolchain): the `.text` section *is*
  the raw A5 instruction stream (after ccec compile + `_link_incore` resolves
  `.rela.text`). Stripping to `.text` and feeding it as a
  `RT_DEV_BINARY_MAGIC_ELF` blob works — simpler has run this path for all
  27 baseline operators.
- **Route 2 `.o`** (VPTO→bisheng fatobj): the device code lives in a
  `__aicore_rel_binary` section that is itself a nested device ELF, and the
  outer `.text` holds **host-side glue** (the host wrapper `rms_norm`, plus
  `rtRegisterGlobals` and `cceModuleCtor`) — not the A5 kernel body.
  `extract_text_section` reads the *outer* `.text` (the host glue), not the
  device code → the blob handed to `rtRegisterAllKernel` has no kernel body
  → launch finds no kernel symbol. (An earlier draft of this doc described
  `.text` as a "1-byte `c3 ret` stub"; that was carried over from a
  *different* artifact, `/tmp/native_rmsnorm.o`, and does not hold for the
  `build_output/vpto_rms_norm/rms_norm.o` fatobj actually cited here — its
  `.text` is 197 bytes of real x86-64 host code. The bug stands regardless:
  `.text` is host code, not device code.)

**Verified fatobj layout** (`readelf -S build_output/vpto_rms_norm/rms_norm.o`):

```
[ 2] .text              PROGBITS  offset 0x040, size 0xc5  ← 197-B host glue (rms_norm wrapper + rtRegisterGlobals + cceModuleCtor)
[ 5] __aicore_rel_binary PROGBITS  offset 0x120             ← nested device ELF (kernel body), begins 7f 45 4c 46
[ 6] __aicore_rel_rec    PROGBITS  offset 0x1b10            ← 24-B descriptor record (CUBE magic) pointing at the device binary; relocations live in .rela.text/.rela__aicore_rel_rec
```

The "fat-object incompatibility" is real, but it sits entirely in the Python
`extract_text_section` step — not in a missing CANN loader.

## 3. Proposed change (two localized Python edits, no C++)

### 3.1 Edit 1 — detect Route-2 fatobjs and feed the right bytes

In `device_runner.py` `compile_and_assemble` (or, cleaner, inside
`elf_parser.extract_text_section`), branch on the object layout:

- **If the `.o` has a `__aicore_rel_binary` section** (bisheng fatobj
  signature) → do NOT strip to `.text`. Feed either:
  - (a) the **whole `.o`** as the blob (let `rtRegisterAllKernel` parse the
    nested ELF via the fat-object loader), OR
  - (b) the **inner device ELF** — the bytes starting at the
    `__aicore_rel_binary` section's `sh_offset` (which begins with
    `7f 45 4c 46` ELF magic) — as the blob.
- **Else** (Route-1 plain `.o`) → keep current `extract_text_section`
  behavior (return `.text`).

**Discriminator** (cheap, robust): walk the ELF section header table (the
parsing code already exists in `_extract_text_elf64`) and check for a section
named `__aicore_rel_binary`. (An earlier draft proposed a secondary "outer
`.text` size implausibly small (≈1 B)" heuristic; that is **unreliable** — the
real `rms_norm.o` fatobj has a 197-byte `.text` of host glue, so the size
heuristic would not fire. Section-name presence is the sole robust signal and
is also absent on Route-1 EmitC `.o`s, so it cleanly gates the new path.)

### 3.2 Edit 2 — extract the inner ELF (fallback if needed)

If CANN's `rtRegisterAllKernel` with `magic=ELF` rejects the whole fatobj
(option 3.1-a), add a helper `extract_aicore_binary(elf_data) -> bytes` to
`elf_parser.py` that:

1. Walks section headers, finds `__aicore_rel_binary`.
2. Returns `elf_data[sh_offset : sh_offset + sh_size]`.
3. Optionally verifies the first 4 bytes are `7f 45 4c 46` (ELF magic).

This is a ~15-line pure-Python function reusing the existing section-walk
machinery in `_extract_text_elf64`. Far simpler than any "Seam 2 /
`resolved_addr_` rework" floated in earlier notes.

### 3.3 Where the edits live

- `pypto/python/pypto/runtime/elf_parser.py` — add `is_bisheng_fatobj(data)`
  + `extract_aicore_binary(data)`; optionally route `extract_text_section`
  through the new branch internally.
- `pypto/python/pypto/runtime/device_runner.py:514` (inside
  `compile_single_kernel`) — change to call a dispatch that picks
  fatobj-aware extraction on HW platforms.
- **C++ unchanged.** `launch_aicore_kernel` already calls `rtRegisterAllKernel`.

Both edits are pure Python, localized to two files, ~30-40 lines total.

## 4. What this unblocks

With the Python loader fixed, simpler's `run_jit(fn=, specs=, golden_fn=)`
flow drives Route 2 end-to-end:

```
golden.run_jit(fn=<@pl.jit module>, specs=build_tensor_specs(...), golden_fn=golden_<name>,
               runtime_cfg=dict(platform="a5", device_id=0))
  → fn.compile()        # pypto frontend → .pto → ptoas VPTO → fatobj .o
  → extract_text_section(fatobj)   ← NOW returns the nested device ELF, not the stub
  → rtRegisterAllKernel(blob)     ← CANN loads the real kernel body
  → rtKernelLaunchWithHandleV2    ← NPU executes
  → validate_golden                ← compare vs torch golden
```

DSV4 modules then get the **same one-command, module-level experience Qwen
has on Route 1**:

```bash
python models/deepseek_v4_pro/rmsnorm.py -p a5 -d 0   # Route 2, whole module
```

Non-leaf kernels run automatically (simpler drives the whole module's
kernel sequence, feeding intermediates between kernels) — no Phase 5
intermediate-capture harness needed. The PMU path
(`pmu_idc_aic_vec_busy_o`, tied to simpler's task/dfx machinery) is
*expected* to work unchanged since simpler's ctypes/module-launch path
already wires PMU capture — but this has **not** been verified on a
Route-2 fatobj load and should be checked in the first end-to-end run.

## 5. Prerequisites partially satisfied (the §5.4 caveat is only half-cleared)

An earlier analysis flagged a scope caveat: the simpler change only unblocks
*loading*; it does not fix ptoas VPTO emitting an empty `kernel.o` for
pypto-emitted `.pto`s. **That half is now satisfied.** The remaining half —
whether `rtRegisterAllKernel` accepts an *unlinked* fatobj — is **not**
cleared and is the real open risk (see §6).

The `vpto-board-validate` skill (commits `879adae`+`3616481`+`deb623e`)
solved the empty-fatobj problem with two mandatory sed preprocessing steps
on the `.pto` IR + the proven ptoas flag set:

- sed 1: module attrs add `pto.kernel_kind = #pto.kernel_kind<...>`
- sed 2: func attrs add `pto.kernel` (triggers kernel body codegen)
- ptoas flags: `--pto-arch=a5 --pto-level=level3 --pto-backend=vpto
  --enable-tile-op-expand --enable-insert-sync --enable-op-fusion`

Verified output: `build_output/vpto_rms_norm/rms_norm.o` is a real fatobj
with a kernel body, the `T rms_norm` kernel symbol, the ctor, and nested ELF
offsets — not the empty ctor-only fatobj of the §5.4 era. (File size is
~9.2 KB as of 2026-08-12; the earlier "8712 bytes" figure is stale and was
not re-verified.)

**Important caveat:** the skill's working path does **not** prove simpler's
`rtRegisterAllKernel` accepts the fatobj. The skill **bypasses simpler
entirely** — it links the fatobj with `bisheng --cce-fatobj-link -shared`
into a `.so` and drives launch through a generated `main.cpp`/`launch.cpp`
using CCE triple-chevron `<<<>>>` launch syntax (which bisheng lowers to a
`__cce_rtKernelLaunch*` call) plus implicit `.so` module-load via
`-l<kernel>_kernel` (SKILL.md L32-34 characterizes this as the
`rtRegisterGlobals` / `__cce_rtKernelLaunchWithFlagV2` path; those API
names appear in SKILL.md prose, not verbatim in the skill's generated
code). So §5.4's "ptoas emits empty kernel.o" caveat is cleared, but the
loader-acceptance question for simpler's `rtRegisterAllKernel` path is
still open and is the load-bearing unknown for this design.

## 6. Effort and risk assessment

### Effort
- **Code: ~30-40 lines** across two Python files. No C++.
- **Testing: 1-2 rounds** on a real A5 to determine whether option 3.1-a
  (whole fatobj) or 3.1-b (inner ELF) is what `rtRegisterAllKernel` accepts.
  This cannot be determined statically — CANN's acceptance of the blob must
  be observed empirically. The design has a built-in fallback (if a fails,
  try b), so the risk is "iterate once", not "architectural rework".

### Risk
- **Low architectural risk for the Route-1 path.** C++ runtime is unchanged.
  Route 1 is preserved (the discriminator gates the new path on
  `__aicore_rel_binary` presence, which Route-1 `.o`s do not have).
- **Unknown: CANN's nested-ELF parsing.** Whether `rtRegisterAllKernel` with
  `magic=ELF` parses a bisheng fatobj as-is (3.1-a) or needs the inner ELF
  (3.1-b) is unverified. The fallback covers both.
- **No regression surface for Route 1.** The `__aicore_rel_binary` check is
  absent on EmitC `.o`s, so they take the existing `extract_text_section`
  path verbatim.

### The load-bearing open risk — relocation / linking (MUST verify first)
The fallback a→b covers *blob format*, but **neither option addresses
unresolved relocations**, and this is the risk that can invalidate the
"two localized Python edits, no C++, ~30-40 lines" estimate:

- The device code section is `__aicore_rel_binary` — the `_rel` suffix marks
  it **relocatable** — and the sibling `__aicore_rel_rec` section is a
  24-byte descriptor (CUBE magic) pointing at the device binary, with the
  actual relocation entries living in `.rela.text` (host-side) and
  `.rela__aicore_rel_rec`.
- The `vpto-board-validate` skill's *working* path runs
  `bisheng --cce-fatobj-link -shared` to produce a `.so` before CANN loads
  it. That link step exists precisely to **resolve the relocations** in
  `__aicore_rel_binary`. It is plausibly load-bearing and non-optional.
- **Route 1 already links, but with a different linker.** Simpler's
  `compile_incore` (a5) calls `_link_incore` (`simpler_setup/kernel_compiler.py:409`),
  which runs the ccec linker (`ld.lld`, `-e kernel_entry`) to resolve
  `.rela.text` — notably the `.bl.uninit.*` block-local globals CANN AscendC
  headers declare (`g_vecTPipePtr`, `g_kfcClient`) — *before*
  `extract_text_section` extracts `.text`. So a link step is **not new
  architecture** for simpler; the question is whether the *existing* ccec
  `ld.lld` path can resolve a bisheng fatobj's *device-side* relocations
  (`__aicore_rel_binary` / `__aicore_rel_rec`), or whether simpler must
  additionally invoke `bisheng --cce-fatobj-link` — a different linker for a
  different object format. (Route 1 incore uses ccec, not g++; the
  earlier "EmitC→g++" shorthand was imprecise.)
- Option 3.1-a (whole fatobj `.o`) and 3.1-b (inner ELF bytes from
  `sh_offset`) both hand CANN a device ELF with **unresolved
  relocations** unless a link step runs first.
- If `rtRegisterAllKernel` rejects an unlinked fatobj (likely, given the
  skill links first), the change is **not** "two Python edits" —
  simpler/pypto's compile path must additionally invoke
  `bisheng --cce-fatobj-link` (ccec's `ld.lld` won't do) on the fatobj
  before feeding it as the blob. That is a compile-pipeline change, not a
  loader tweak. The silver lining: simpler already has `_link_incore`
  scaffolding, so the plumbing site exists — but it is ccec-specific today.

**Prerequisite experiment (do before committing to the §3 plan):** on a real
A5, call `rtRegisterAllKernel` directly with (a) the whole `rms_norm.o`
fatobj, (b) the inner device ELF bytes from `__aicore_rel_binary`, and
(c) the `bisheng --cce-fatobj-link`-produced `.so`'s relevant bytes, and
observe which (if any) CANN accepts and launches. Only if (a) or (b) works
is the "two localized Python edits" estimate sound. If only (c) works, the
plan must grow a `bisheng --cce-fatobj-link` step in the compile path
(ccec's existing `_link_incore`/`ld.lld` won't do for bisheng fatobjs) and
§6's effort/risk must be re-baselined accordingly.

### What it does NOT fix
- Per-kernel VPTO lowering precision issues (e.g. rms_norm max_diff=30464).
  These remain findings, not regressions introduced by this change — the
  same fatobj is loaded, just via simpler instead of the per-kernel harness.
- The `vpto-board-validate` skill remains useful for kernel-level isolation
  and debugging; this change adds a module-level path, it does not replace
  the skill.

## 7. Relationship to current work

This is a **separate, parallel workstream** from the `tests/dsv4_validate`
sweep framework (commit `8c5b6dd` and successors). That framework runs
Route 2 via the per-kernel `vpto_run.py` skill harness. This simpler change
would add a Route-2-via-simpler path that the framework could later invoke as
an alternative `--route` mode, giving module-level coverage that the
kernel-level skill cannot provide. They are complementary, not conflicting.

## 8. Open questions for the reviewer

1. Is modifying `pypto/python/pypto/runtime/{device_runner.py,elf_parser.py}`
   acceptable? These are upstream pypto files (the framework, not pypto-lib).
   The changes are backward-compatible (Route 1 unaffected) but they do
   touch the framework itself, not just pypto-lib. Note the asymmetry: this
   plan deliberately leaves simpler (also an upstream submodule at
   `runtime/`, pin `3165cc89`) *unmodified* — the C++ claim is "simpler
   unchanged", while the Python claim is "pypto changed". Both are upstream;
   if only pypto-lib-local changes are in scope, the edits need to land via a
   pypto PR (or a pypto-lib-side shim/wrapper) rather than a direct edit.
2. Is there a preference for option 3.1-a (feed whole fatobj) vs 3.1-b
   (extract inner ELF)? If unknown, the implementation should try a, fall
   back to b — confirm this is the desired default behavior.
3. Should this be gated behind a flag (e.g. `--vpto-fatobj-loader`) so the
   behavior is opt-in until validated, or applied unconditionally (since
   Route 1 is gated by the `__aicore_rel_binary` discriminator)?

## 9. Proven ptoas invocation (for the compile step, unchanged)

For reference, the ptoas + bisheng sequence that produces a valid Route-2
fatobj (the skill already does this; simpler's `fn.compile()` would need to
emit equivalent flags, OR the skill's fatobj can be fed to simpler directly
as a pre-compiled `.o`):

```bash
# sed-preprocess .pto (add pto.kernel_kind + pto.kernel attrs)
# ptoas VPTO lowering
ptoas --pto-arch=a5 --pto-level=level3 --pto-backend=vpto \
      --enable-tile-op-expand --enable-insert-sync --enable-op-fusion \
      <kernel>.pto -o <kernel>.o   # ← the fatobj this change loads
```

(Whether simpler's `fn.compile()` calls ptoas with these flags, or whether
the fatobj is produced externally and handed to simpler, is an integration
detail to resolve during implementation — see open question 1.)

## 10. Empirical verdict — the prerequisite experiment was run (2026-08-12)

The §6 "prerequisite experiment" was executed on a real Ascend950PR (device
1, CANN 9.2.0). A minimal probe (`build_output/vpto_probe/route2_probe.cpp`)
mirrored simpler's `DeviceRunnerBase::launch_aicore_kernel` setup
(`rtSetDevice` → `rtStreamCreate` → `rtRegisterAllKernel` via `dlopen` on
`libruntime.so`) and fed it the `rms_norm` fatobj three ways.

### 10.1 Phase 1 — registration acceptance (VERIFIED, surprising)

Both Route-2 blobs were **accepted** by `rtRegisterAllKernel`:

| blob | bytes | rc | handle | result |
|------|-------|----|--------|--------|
| (a) whole fatobj `.o` | 9080 | 0 | non-null | **ACCEPTED** |
| (b) inner device ELF (`__aicore_rel_binary` @0x110, 0x1971 B) | 6513 | 0 | non-null | **ACCEPTED** |

So the §6 relocation/linking risk **did not materialize at registration
time**. CANN's `rtRegisterAllKernel` parses the bisheng fatobj (or its inner
device ELF) without requiring `bisheng --cce-fatobj-link` first. The §6 worry
("CANN rejects unlinked fatobj because `rtDevBinary_t` has no relocation
fields") is empirically **false for this toolchain version**. Option 3.1-a
and 3.1-b both work as loader inputs.

### 10.2 Phase 2 — the launch ABI is the real blocker (VERIFIED, decisive)

Registration succeeding is **not** sufficient, because simpler's launch
path and the VPTO fatobj's kernel entry use **incompatible ABIs**. This is
the load-bearing finding; it invalidates the §3 "two localized Python edits"
thesis.

**Route 1 (simpler) blob composition — what `.text` actually contains:**

The blob fed to `rtRegisterAllKernel` on Route 1 is **not** the user kernel
alone. simpler's compile path (`compile_incore`, `kernel_compiler.py:313`)
links **three** pieces with `ld.lld -e kernel_entry` into one `.text`:

```
KERNEL_ENTRY(aicore_kernel)   ← a5/platform/onboard/aicore/kernel.cpp:104
  │  reads __gm__ KernelArgs *k_args  (the framework struct from rtArgsEx_t.args)
  │  publishes per-core profiling slots (PMU, L2 swimlane)
  └→ aicore_execute(runtime_args, block_idx, core_type)
     │  ← user-side hook, defined in the generated kernel source
     └→ kernel_entry(__gm__ int64_t* args)   ← pto_backend.py:_generate_kernel_wrapper:764
        │  unpacks tensor ptrs from the int64_t* args buffer
        └→ user kernel (rmsnorm / scatter_softmax_pool / ...)
```

- `rtKernelLaunchWithHandleV2` passes `rtArgsEx_t.args` pointing at a
  `KernelArgs` struct (`kernel_args.h:79`: `runtime_args`, `regs`,
  `pmu_data_base`, … — framework fields, **not** tensor pointers).
- `KERNEL_ENTRY(aicore_kernel)` consumes that struct and dispatches into
  `aicore_execute`, which calls `kernel_entry`, which unpacks tensors.

So the **entry symbol** the blob must expose is `kernel_entry` (or
`aicore_kernel` via the stub), and the **arg contract** is a single
`__gm__ int64_t*`/`__gm__ KernelArgs*` — not a flat tensor-pointer list.

**Route 2 (VPTO) fatobj composition — verified symbol table:**

```
readelf -s rms_norm.o:
  4: ...  28 FUNC LOCAL  DEFAULT 2 rtRegisterGlobals
  5: ...  21 FUNC LOCAL  DEFAULT 2 cceModuleCtor
  9: ... 137 FUNC GLOBAL DEFAULT 2 rms_norm
 10: ...  0  NOTYPE  GLOBAL DEFAULT UND __cce_rtKernelLa[...]   (LaunchWithFlagV2)
 12: ...  0  NOTYPE  GLOBAL DEFAULT UND rtFunctionRegister
```

There is **no `kernel_entry`**, **no `KERNEL_ENTRY(aicore_kernel)` stub**,
and **no `aicore_execute`**. The fatobj's kernel is entered via the skill's
generated `launch.cpp`:

```cpp
extern "C" __global__ AICORE void rmsnorm(__gm__ bfloat16_t* v1,
                                          __gm__ bfloat16_t* v2,
                                          __gm__ float* v3);
void LaunchRmsnorm(uint16_t *v1, uint16_t *v2, float *v3, void *stream) {
    rmsnorm<<<1, nullptr, stream>>>((__gm__ bfloat16_t*)v1, ...);
}
```

i.e. the kernel reads **flat `__gm__` tensor pointers** as direct
parameters, driven by bisheng's `<<<>>>` lowering to
`__cce_rtKernelLaunchWithFlagV2`. This is the **only** entry contract the
fatobj's `rms_norm` understands.

### 10.3 The ABI mismatch, precisely

| | Route 1 (simpler) | Route 2 (VPTO fatobj) |
|---|---|---|
| entry symbol exposed | `kernel_entry` / `aicore_kernel` stub | `rms_norm` (direct `__global__ AICORE`) |
| arg contract | `KernelArgs*` → `int64_t*` (framework unpacks tensors) | flat `__gm__` tensor ptrs as direct kernel params |
| launch API | `rtKernelLaunchWithHandleV2(handle, …, rt_args=KernelArgs)` | `<<<1, nullptr, stream>>>(ptrs)` → `__cce_rtKernelLaunchWithFlagV2` |
| module-load | `rtRegisterAllKernel` (blob = linked `.text`) | `.so` + `rtRegisterGlobals`/ctor (skill path) |

Feeding the Route-2 fatobj to `rtRegisterAllKernel` (which Phase 1 showed
succeeds) produces a handle, but calling `rtKernelLaunchWithHandleV2` on it
would jump to offset 0 of the fatobj's device ELF — which is **`rms_norm`'s
prologue expecting three `__gm__` tensor pointers**, not a `KernelArgs*`.
The args buffer simpler builds (`KernelArgs` with `runtime_args` at offset
0) would be reinterpreted as the first tensor pointer → garbage pointer →
fault. There is no `kernel_entry`/`aicore_execute` trampoline in the fatobj
to bridge the two contracts.

### 10.4 Verdict — which route is viable

**The §3 plan (Python-only loader edit, no C++/ABI work) is NOT viable.**
Registration acceptance was the easy part and it passed. The hard part is
that the Route-2 fatobj and simpler's `rtKernelLaunchWithHandleV2` speak
different entry/arg ABIs. Making Route 2 run through simpler requires one of:

1. **Link a simpler-style stub into the Route-2 fatobj.** Generate a
   Route-2-aware `KERNEL_ENTRY(aicore_kernel)` + `aicore_execute` +
   `kernel_entry(int64_t*)` trampoline (the Route-1 dispatch chain) and link
   it with the ptoas VPTO fatobj via `bisheng --cce-fatobj-link`, so the
   resulting blob exposes `kernel_entry` and unpacks `KernelArgs`/`int64_t*`
   into the flat `__gm__` ptrs `rms_norm` expects. This is a **codegen +
   link-pipeline change in pypto/simpler**, not a loader edit. Effort:
   medium-large; touches `_generate_kernel_wrapper` / `compile_incore` /
   a new bisheng-link path alongside ccec's `ld.lld`.

2. **Add a second launch path to simpler that uses the fatobj's native ABI.**
   Teach simpler to, for Route-2 fatobjs, skip `KERNEL_ENTRY`/`KernelArgs`
   and instead drive `<<<>>>`/`__cce_rtKernelLaunchWithFlagV2` with flat
   tensor pointers (what the skill does today). This is an **architectural
   addition** — simpler would carry two launch ABIs (Route-1 `KernelArgs`,
   Route-2 flat-ptrs). Effort: large; new C++ in `device_runner_base.cpp`,
   new Python dispatch in `device_runner.py`, plus the `.so`/dlopen plumbing
   the skill currently owns.

3. **Keep the per-kernel skill as the Route-2 path.** Accept that Route 2
   does not get simpler's module-level `run_jit` experience; keep extending
   `vpto-board-validate` (Phase 5 intermediate capture for multi-kernel
   modules). Effort: zero new architecture; the status quo.

**Recommendation:** Option 1 is the most faithful to the "one runtime, two
backends" goal and reuses simpler's existing dispatch chain — but it is a
codegen/link change, definitively **not** the "~30-40 lines of Python" the
§3/§6 estimate assumed. Option 3 is the honest status quo. Option 2 is the
heaviest and least aligned with simpler's design. The §3 "two localized
Python edits" plan should be **withdrawn** in favor of a new plan sized to
Option 1 (or deferred to Option 3) pending a scope decision from the owner.

### 10.5 Artifacts produced

- `build_output/vpto_probe/route2_probe.cpp` + `route2_probe` (the Phase-1
  registration probe; `dlopen` libruntime, mirrors `launch_aicore_kernel`
  setup, stops at registration). Re-runnable:
  `LD_LIBRARY_PATH=<cann>/x86_64-linux/lib64 ./route2_probe 1 a-whole-fatobj <fatobj> 0 0`
  and `./route2_probe 1 b-inner-elf <fatobj> 0x110 0x1971`.
- Captured output: both blobs return `rc=0` + non-null handle on device 1.

### 10.6 Minor corrections surfaced by the experiment

- `__aicore_rel_binary` section offset is `0x110` (size `0x1971`), not
  `0x120` as the §2.2 readelf snippet implied. The snippet's offset was
  approximate; the experiment used the parsed `0x110`.
- `.text` is `0xc5` (197 B), already corrected in §2.2.
- `libruntime.so` (CANN 9.2.0) exports `rtRegisterAllKernel` (`T` @0x37090),
  `rtKernelLaunchWithHandleV2` (@0x59040), `rtSetDevice` (@0x60710) — all
  live and dlsym-able, confirming simpler's path is the public CANN API.

## 11. Option-1 link experiment — the object-format wall (2026-08-12)

§10.4 named three routes. Option 1 ("link a simpler-style stub into the
Route-2 fatobj") was the one most aligned with the "Route 2 should behave
like Route 1" goal, so it was tested empirically next. **It is NOT viable.**
The blocker is not ABI or codegen effort — it is a hard object-format
incompatibility between the two toolchains.

### 11.1 What the experiment did

Goal: produce a single loadable blob whose `.text` offset 0 is a
simpler-contract `kernel_entry(__gm__ int64_t* args)` that unpacks tensor
ptrs and forwards to the Route-2 `rms_norm`, then feed that blob to
`rtRegisterAllKernel` + `rtKernelLaunchWithHandleV2`.

Steps:
1. Wrote `bridge_ccec.cpp` — a `void kernel_entry(__gm__ int64_t* args)`
   that reinterprets `args[0..2]` as `__gm__ bf16/bf16/float` ptrs, reads
   `args[3..4]` as SPMD `block_idx/block_num` (defaulting 0/1), and calls
   `rms_norm(v1,v2,v3,block_idx,block_num)`. (`rms_norm.pto` confirmed the
   5-param signature: 3 bf16 ptrs + 2 i32 SPMD scalars; the skill's 3-param
   `<<<1>>>` call auto-defaults the scalars, and `rms_norm` standalone
   PASSES on board — `PRECISION_REPORT.md` — so the 5-param call with 0/1
   is correct for a single-block launch.)
2. Compiled it to a device `.o` — tried both Route-1's compiler (ccec) and
   bisheng in CCE-device mode (`bisheng -x cce --cce-aicore-only`).

### 11.2 Result — bridge compiles fine, but linking is impossible

**Bridge compilation succeeded under both toolchains:**

| compiler | command | output | kernel_entry | Machine |
|----------|---------|--------|--------------|---------|
| ccec | `ccec -x cce --cce-aicore-only --cce-aicore-arch=dav-c310-vec` | `bridge_ccec.o` | 92 B @ `.text` offset 0, GLOBAL | `0x1029` (CCE device) |
| bisheng | `bisheng -x cce --cce-aicore-only --cce-aicore-arch=dav-c310-vec` | `bridge_bisheng_cce.o` | 108 B @ `.text` offset 0, GLOBAL | `0x1029` (CCE device) |

Both produce a clean CCE device `.o` with `kernel_entry` at `.text` offset 0
and `rms_norm` as UND — exactly the Route-1 pre-link layout. So the
**stub codegen is not the blocker**; the bridge trampoline is trivially
producible.

**Linking the bridge with the Route-2 fatobj failed in every direction:**

| linker | inputs | result |
|--------|--------|--------|
| ccec `ld.lld -e kernel_entry` | bridge_ccec.o + rms_norm.o | `error: rms_norm.o is incompatible with bridge_ccec.o` |
| bisheng `--cce-fatobj-link -shared` | bridge_ccec.o + rms_norm.o | `error: bridge_ccec.o is incompatible with elf64-x86-64` |
| bisheng `--cce-fatobj-link -shared` | bridge_bisheng_cce.o + rms_norm.o | `error: bridge_bisheng_cce.o is incompatible with elf64-x86-64` |

### 11.3 Root cause — two mutually-foreign ELF object formats

```
bridge (ccec or bisheng -x cce):   Machine = 0x1029 (CCE device), .text = device code
Route-2 fatobj (bisheng default):  Machine = x86-64,           device code nested in __aicore_rel_binary
```

These are **different ELF dialects**, not just different sections:
- The CCE device `.o` is `e_machine = 0x1029` with device instructions
  directly in `.text`. ccec's `ld.lld` links these.
- The bisheng fatobj is `e_machine = EM_X86_64` (a host relocatable) whose
  device code lives in a nested ELF under `__aicore_rel_binary`. bisheng's
  `--cce-fatobj-link` (which drives an x86-64 `ld.lld`) links these.

ccec's `ld.lld` (targets `elf64-cce`) rejects the x86-64 fatobj as
"incompatible"; bisheng's `--cce-fatobj-link` (targets `elf64-x86-64`)
rejects the CCE device `.o` as "incompatible with elf64-x86-64". **Neither
linker accepts the other's object type.** There is no linker in the toolchain
that consumes both a CCE device `.o` and a bisheng x86-64 fatobj in one link.

### 11.4 What this means for Option 1

Option 1 as stated ("link a simpler-style stub into the Route-2 fatobj") is
**dead at the object-format level**, not merely "expensive". You cannot
produce a single blob whose `.text` offset 0 is a ccec/bisheng-CCE
`kernel_entry` and whose body also contains the bisheng-fatobj `rms_norm`,
because no linker will combine a CCE device `.o` with a bisheng x86-64
fatobj. The stub compiles; it just can't be merged with the fatobj.

The only ways around the object-format wall would be:
- **(1')** Teach ptoas/bisheng to emit the simpler-contract `kernel_entry`
  *inside* the fatobj's `__aicore_rel_binary` (as bisheng device code, not a
  separate CCE `.o`). This is a ptoas VPTO codegen change — emit an extra
  device function in the same nested ELF that ptoas already produces. This is
  the real Option-1-shaped path, but it is a ptoas change, not a "link a stub".
- **(1'')** Teach simpler to, for Route-2, read the fatobj's
  `__aicore_rel_binary` nested ELF and register *that* as the blob (Phase 1 of
  §10 showed `rtRegisterAllKernel` accepts it), then drive a flat-ptr launch
  (`<<<>>>`/`__cce_rtKernelLaunchWithFlagV2`) instead of
  `rtKernelLaunchWithHandleV2`. But this is Option 2 (the second launch path),
  not Option 1.

### 11.5 Revised route assessment

Given §10's ABI mismatch AND §11's object-format wall:

| route | status after experiment |
|-------|------------------------|
| §3 Python-only loader edit | **dead** (§10 ABI mismatch) |
| Option 1 (link simpler stub into fatobj) | **dead** (§11 object-format wall) |
| Option 1' (ptoas emits kernel_entry inside the fatobj's nested ELF) | **viable, ptoas codegen change** — the real Option-1 path. Effort: medium; touches ptoas VPTO lowering, not simpler/pypto loader. |
| Option 2 (simpler learns flat-ptr launch + `__aicore_rel_binary` extraction) | **viable, simpler C++ + pypto change** — the §10.4 option 2. Effort: large; new launch ABI in simpler + the `.so`/dlopen plumbing the skill owns. |
| Option 3 (keep per-kernel skill) | **zero-effort status quo** |

**For the owner's stated goal** ("Route 2 like Route 1, same-granularity
board run, shared golden") the viable paths are now **Option 1'** (make
ptoas emit the simpler-contract entry inside the fatobj) or **Option 2**
(make simpler speak the fatobj's native flat-ptr launch). Option 1' is the
smaller codegen change and keeps simpler/golden unchanged; Option 2 is the
"one runtime speaks two ABIs" architectural change. A scope decision between
1' and 2 is now the gating question.

### 11.6 Artifacts produced

- `build_output/vpto_probe/bridge_ccec.cpp` — the simpler-contract bridge.
- `build_output/vpto_probe/bridge_ccec.o` — ccec-compiled (Machine 0x1029,
  kernel_entry @ .text offset 0, 92 B). Links standalone; refuses the fatobj.
- `build_output/vpto_probe/bridge_bisheng_cce.o` — bisheng `-x cce
  --cce-aicore-only`-compiled (Machine 0x1029, kernel_entry @ .text offset 0,
  108 B). Also refuses the fatobj at link.
- All three link attempts (ccec ld.lld, bisheng --cce-fatobj-link ×2) logged
  "incompatible" errors, reproduced by the commands in §11.2.

## 12. The offset-0 question resolved — CANN resolves by name, not offset (2026-08-12)

§11.4 flagged one open risk for Option 1': simpler's `rtKernelLaunchWithHandleV2`
was believed (per simpler's own `_link_incore` docstring) to "jump to offset 0",
which would force `kernel_entry` to be placed first in the blob — something ptoas
does not control. This was investigated against CANN headers + simpler's actual
runtime code + the skill's `.so` disassembly. **The risk is cleared: CANN
resolves kernel entry by symbol name / tilingKey, NOT by jumping to offset 0.**
The "offset 0" claim is simpler's *own InCore-loader convention*, which Route 2
bypasses entirely.

### 12.1 Evidence — CANN's resolution model

- `rtDevBinary_t` (`rt_external_kernel.h:85-90`) is `{magic, version, data,
  length}` — a dumb descriptor. But the blob CANN receives is a **complete
  ELF**, not a raw `.text` slice. The skill's `librms_norm_kernel.so` embeds
  the device ELF in a section beginning with `7f 45 4c 46` (`\x7fELF`),
  confirmed by `readelf -x .aicore_binary`. So CANN *parses* the ELF and
  indexes kernels internally.
- CANN headers explicitly describe a multi-kernel-per-blob model with
  selector-based lookup:
  - `rtRegisterAllKernel` (`rt_external_kernel.h:551-557`): "register device
    binary with **all kernel**" (plural) — one blob can hold many kernels.
  - `rtBinaryGetFunction(binHandle, tilingKey, &funcHandle)` (`kernel.h:748-757`):
    "Find funcHandle based on binHandle and **tilingKey**." A selector only
    makes sense if one handle maps to multiple kernels.
  - `rtBinaryGetFunctionByName(binHandle, kernelName, &funcHandle)`
    (`kernel.h:878-886`): "Find funcHandle based on binHandle and
    **kernelName**." Name-based lookup within one registered binary.
  - `rtKernelInfo` (`kernel.h:65-73`) has a `task_offset` field ("kernel offset
    in module") — CANN **computes** a per-kernel offset from the ELF's symbol
    table; it does not assume offset 0.
  - `rts_kernel.h:85`: `rtsBinaryUnload` "Will unregister **all kernels the
    binary contains**" — direct statement that one binary contains multiple
    kernels.
  - `rts_kernel.h:96-112`: `rtsFuncGetByName(binHandle, kernelName, ...)` and
    `rtsFuncGetByEntry(binHandle, funcEntry, ...)` — name- and numeric-entry
    selectors within a binary.

### 12.2 Evidence — simpler's actual call confirms tilingKey is a selector

`device_runner_base.cpp:1242-1267`:
```cpp
binary.magic = RT_DEV_BINARY_MAGIC_ELF;       // whole-ELF tag, not a .text slice
binary.data = aicore_kernel_binary_.data();   // the whole ELF
binary.length = aicore_kernel_binary_.size();
int rc = rtRegisterAllKernel(&binary, &aicore_bin_handle_);
...
int rc = rtKernelLaunchWithHandleV2(aicore_bin_handle_, 0, block_dim_, &rt_args, nullptr, stream, &cfg);
//                                                          ^ tilingKey hardcoded 0
```
The blob is the **whole ELF** (`magic=RT_DEV_BINARY_MAGIC_ELF`), and
`tilingKey=0` is a *selector* (simpler registers one-kernel blobs and asks for
tilingKey 0). The existence of `rtBinaryGetFunction(tilingKey)` and
`rtBinaryGetFunctionByName(kernelName)` proves the handle resolves to a specific
kernel by selector, not by jumping to offset 0.

### 12.3 Evidence — the skill's `.so` proves the name-based path end-to-end

Disassembly of `librms_norm_kernel.so`:
```
0x5870: mov  0x2239(%rip),%rsi   # stubFunc  (from .data)
0x5877: lea  -0x4bf2(%rip),%rcx  # .rodata:0xc8c -> the string "rms_norm"
0x5884: mov  %rcx,%rdx           # stubName = "rms_norm"
0x5887: jmp  rtFunctionRegister@plt
...
0x65ff: call rtKernelLaunchWithFlagV2@plt   # launch with the registered stubFunc
```
`rtFunctionRegister(binHandle, stubFunc, stubName="rms_norm", ...)` binds the
**name** `"rms_norm"` to the device entry; `rtKernelLaunchWithFlagV2(stubFunc,
...)` resolves by that binding. The string `"rms_norm"` is embedded in
`.rodata`. **No offset-0 assumption anywhere in the skill's path.**

### 12.4 The "offset 0" claim is simpler's InCore-only convention

simpler's `kernel_compiler.py:411-414` and `elf_parser.py:55-57`:
```python
# The AICore loader jumps to offset 0 of the payload, so this symbol must be
# the first thing in .text.
_ENTRY_SYMBOL = "kernel_entry"
```
This lives in `simpler_setup/` and describes simpler's **own InCore loader**
(`extract_text_section` strips a raw `.text` slice and simpler's runtime jumps
to offset 0 of *that slice*). The `vpto-board-validate` skill **explicitly
bypasses** this InCore path (SKILL.md:30-34): it uses CANN's ELF module-load
(`rtFunctionRegister` / `rtKernelLaunchWithFlagV2`), which is name-based.
simpler's onboard `launch_aicore_kernel` *also* uses the whole-ELF +
`rtRegisterAllKernel` path (§12.2), not the raw-`.text` InCore path.

So the two resolution models are:

| | simpler InCore (legacy, raw `.text`) | CANN ELF path (Route 2 + simpler onboard) |
|---|---|---|
| Blob | raw `.text` bytes (no ELF) | whole ELF (`RT_DEV_BINARY_MAGIC_ELF`) |
| Entry | offset 0 of the `.text` slice | by **stubName / tilingKey** (CANN parses the ELF) |
| Used by | simpler's `_link_incore` (bypassed by Route 2) | Route 2 skill + simpler's `launch_aicore_kernel` |

### 12.5 Implication for Option 1'

**Offset-0 ordering is NOT a blocker for Option 1'.** Because Route 2 resolves
by symbol name (via `rtFunctionRegister` + `rtKernelLaunchWithFlagV2`, or
equivalently `rtRegisterAllKernel` + `rtBinaryGetFunctionByName`), a fatobj
containing both `kernel_entry` and `rms_norm` at any relative offset will
resolve correctly as long as both are named symbols in the nested ELF's symbol
table. `kernel_entry` does **not** need to be at offset 0.

This means Option 1' (ptoas emits a synthetic `kernel_entry` into the fatobj's
`__aicore_rel_binary`) is now **unblocked on all fronts**:
- ptoas can inject a `func::FuncOp` in the VPTO path (~80-150 lines, §11.4/1'),
- the function lands in the same nested ELF as `rms_norm` (no cross-format
  linking, §11's object-format wall avoided),
- the skill still resolves `rms_norm` by name (§11 confirmed, §12.3 reconfirmed),
- CANN resolves whichever kernel by name/tilingKey regardless of offset
  (§12.1-12.4), so `kernel_entry`'s position in the blob is irrelevant.

### 12.6 Remaining work to fully validate Option 1' (not yet done)

- **Empirical end-to-end**: the §10/§11/§12 evidence is now circumstantially
  complete but has NOT been validated by a single fatobj that actually carries
  both `kernel_entry` + `rms_norm` and is launched through simpler's
  `rtKernelLaunchWithHandleV2`. The cleanest validation is: (1) patch ptoas to
  emit `kernel_entry`, (2) regenerate the fatobj, (3) register it via
  `rtRegisterAllKernel` + `rtFunctionRegister("kernel_entry", ...)` and launch
  via `rtKernelLaunchWithHandleV2(handle, tilingKey=0, ...)` with a packed
  `int64_t args[]` buffer, (4) compare output to the skill's golden.
- **The `kernel_entry` body** must unpack `args[0..N]` into the tensor ptrs
  `rms_norm` expects. Whether this unpacking uses raw `__gm__` ptrs (the
  simplified bridge in §11) or simpler's full `Tensor*` struct (§10.2) depends
  on how simpler's host side packs `rt_args.args` — which is the `KernelArgs`
  struct, not a raw ptr array. So the real `kernel_entry` body must match
  whatever simpler actually puts in the args buffer. This is a design detail to
  resolve during implementation, but it is now a *bounded* one (no
  architectural blocker).
- **simpler's host side** must call `rtFunctionRegister("kernel_entry", ...)`
  (or rely on `rtRegisterAllKernel`'s name indexing) to bind the name before
  `rtKernelLaunchWithHandleV2`. simpler currently passes `tilingKey=0` and
  expects a single-kernel blob; whether it needs a small change to register
  the specific kernel name, or whether `rtRegisterAllKernel` already indexes by
  name for `tilingKey=0`, is the one simpler-side detail to confirm.

### 12.7 Revised verdict

| route | status |
|-------|--------|
| §3 Python-only loader edit | dead (§10 ABI) |
| Option 1 (link external stub into fatobj) | dead (§11 object-format wall) |
| **Option 1' (ptoas emits `kernel_entry` inside the fatobj)** | **viable — unblocked on all fronts.** ptoas codegen change (~80-150 lines, VPTO-only), offset-0 irrelevant (§12), skill path untouched (§11/§12.3). |
| Option 2 (simpler learns flat-ptr launch) | viable but larger (simpler C++ + pypto, second launch ABI) |
| Option 3 (keep per-kernel skill) | zero-effort status quo |

**Option 1' is the recommended path for the owner's stated goal** ("Route 2
like Route 1, same-granularity board run, shared golden"). It is the only
option that achieves "Route 2 behaves like Route 1 at the entry/ABI level"
without adding a second launch ABI to simpler or touching the golden. The
remaining work (§12.6) is implementation, not architecture.

## 13. Option 1' ALSO dead — the raw-.text precondition (2026-08-12, correction of §12)

**§12 was wrong.** When the implementation was attempted, the first step —
running simpler's `extract_text_section` on the Route-2 fatobj — **rejected
the fatobj outright**:

```
ValueError: AICore loader cannot extract a runnable payload from <bytes>:
it contains out-of-line code or relocations against .text that linking did
not resolve (see issue #900).
Unresolved relocations against .text:
  .rela.text  (9 entries)
```

This exposed a conflation in §12: **the "CANN resolves by name" finding
applies to the skill's `.so` path, NOT to simpler's `rtRegisterAllKernel`
path.** The two are genuinely different, and simpler's path cannot consume a
Route-2 fatobj for a reason independent of offset-0 ordering:

### 13.1 The two paths, precisely

| | simpler onboard path | skill Route-2 path |
|---|---|---|
| Compile | `ccec -x cce --cce-aicore-only` → device `.o` (Machine 0x1029, pure `.text`) | `bisheng` → x86-64 fatobj (device in `__aicore_rel_binary`, nested ELF) |
| Link | `ld.lld -e kernel_entry` → **resolves `.rela.text`** → flat `.text` with `kernel_entry` at offset 0 | `bisheng --cce-fatobj-link -shared` → `.so` (host ELF, device nested) |
| Extraction | `extract_text_section(linked.o)` → raw `.text` bytes (linkage already resolved) | (none — `.so` is the loadable unit) |
| Blob fed to CANN | raw `.text` bytes + `magic=RT_DEV_BINARY_MAGIC_ELF` (the "CUBE" tag) | whole `.so` via CANN module-load |
| Entry resolution | **offset 0** of the raw `.text` (simpler's own convention) | **by name** via `rtFunctionRegister` + ctor |
| Pre-link relocations | **resolved** by `ld.lld` before extraction | **resolved** by `bisheng --cce-fatobj-link` into the `.so` |

### 13.2 Why simpler's path cannot consume the Route-2 fatobj

simpler's `extract_text_section` (`simpler_setup/elf_parser.py`) is the gate.
It is not just "grab `.text`" — it **verifies that `.text` carries no
unresolved relocations** and raises `ValueError` if any exist (issue #900,
historically PR #830 / #831). The Route-2 fatobj is an **unlinked
relocatable**: its outer `.text` is the host glue (`rms_norm` wrapper +
`rtRegisterGlobals` + `cceModuleCtor`) carrying 9 `.rela.text` entries that
only `bisheng --cce-fatobj-link` can resolve. simpler's `ld.lld` cannot
link the bisheng fatobj (§11's object-format wall), so those relocations
stay unresolved, and `extract_text_section` rejects the blob.

This is **independent of offset-0 ordering**. Even if ptoas emitted a
`kernel_entry` into the fatobj, simpler could never get past
`extract_text_section` to even reach `rtRegisterAllKernel`. The fatobj is
not a "pre-linked flat `.text`"; it is a relocatable that requires the very
`bisheng --cce-fatobj-link` step that simpler does not perform.

### 13.3 §12's error, precisely

§12.2 read simpler's `launch_aicore_kernel` as feeding a "whole ELF" to
`rtRegisterAllKernel` (because `magic = RT_DEV_BINARY_MAGIC_ELF`). But
`RT_DEV_BINARY_MAGIC_ELF = 0x43554245` is the ASCII `"CUBE"` **kernel-type
tag**, not a real ELF-magic assertion (§2.1 of this doc, confirmed against
`rt_external_kernel.h:65`). The blob in `aicore_kernel_binary_` is whatever
`extract_text_section` returned — which, for Route-1, is a **flat linked
`.text`**, not a parseable ELF. So CANN receives raw bytes tagged "CUBE" and
jumps to offset 0; it does not parse an ELF symbol table. The
`rtBinaryGetFunctionByName` / `rtBinaryGetFunction(tilingKey)` APIs that §12
cited as evidence of name-based resolution are real CANN APIs, but they
belong to the **`.so` module-load path** (where CANN does parse an ELF), not
the raw-bytes `rtRegisterAllKernel` path that simpler uses. §12 conflated the
two.

### 13.4 The corrected verdict

All four "Option 1 family" paths are now dead or require deeper change:

| route | status |
|-------|--------|
| §3 Python-only loader edit | dead (§10 ABI) |
| Option 1 (link external ccec stub into fatobj) | dead (§11 object-format wall) |
| Option 1' (ptoas emits `kernel_entry` into fatobj) | **dead (§13 — simpler's `extract_text_section` rejects the unlinked fatobj; the raw-`.text` precondition is unmet)** |
| Option 1'' (ptoas emits `kernel_entry` AND simpler learns to link the fatobj) | subsumed by Option 2 |
| **Option 2 (simpler learns the `.so`/flat-ptr path)** | **the only viable path that reaches simpler.** Requires simpler to, for Route-2, (a) extract the `__aicore_rel_binary` nested ELF or link the fatobj into a `.so`, and (b) drive launch via `rtFunctionRegister`+`rtKernelLaunchWithFlagV2` (name-based) instead of `rtRegisterAllKernel`+`rtKernelLaunchWithHandleV2` (offset-0). This is the skill's path, brought into simpler. |
| Option 3 (keep per-kernel skill) | zero-effort status quo |

**For the owner's stated goal**, the honest answer is now: **Option 2 or
Option 3.** Option 1' does not work — simpler's onboard path fundamentally
expects a pre-linked flat `.text`, and the Route-2 fatobj is a relocatable
that requires a link step simpler does not perform. The only way to make
Route 2 run through simpler is to teach simpler the skill's `.so`/name-based
path (Option 2), or keep using the skill per-kernel (Option 3).

### 13.5 What "Option 2" actually entails (sized, since it's now the only path)

Option 2 = "simpler speaks the fatobj's native launch path." Concretely:
1. **simpler compile path**: for Route-2 fatobjs, run `bisheng
   --cce-fatobj-link` (the skill's link step) instead of ccec
   `_link_incore`/`extract_text_section`. This produces a `.so`.
2. **simpler launch path**: for Route-2, drive `rtFunctionRegister(name)` +
   `rtKernelLaunchWithFlagV2(stubFunc, flat-tensor-ptrs)` instead of
   `rtRegisterAllKernel(raw .text)` + `rtKernelLaunchWithHandleV2(KernelArgs)`.
   This is a second launch ABI in simpler's `device_runner_base.cpp`.
3. **args packing**: simpler's current `KernelArgs` (framework struct) must
   be bypassed for Route-2; instead pack flat tensor pointers (the skill's
   ABI). simpler's `run_jit` → `CoreCallable.build` path needs a Route-2
   branch that skips `KernelArgs` and builds a flat-ptr args buffer.
4. **module-level**: simpler's resident-kernel dispatch (AICPU → AICore task
   queue → `kernel_entry`) does not apply on the flat-ptr path. To get
   module-level (non-leaf kernels, intermediate tensor feeding) on Route 2,
   simpler would need a host-side module orchestrator (allocate intermediates,
   launch A→B→C in sequence) — the skill does not have this either. So Option
   2 gives per-kernel Route-2-in-simpler first; module-level is a further
   build on top.

Effort: **large** — new C++ launch ABI in simpler + pypto dispatch branch +
bisheng-link integration. It is the "one runtime, two launch ABIs"
architectural change. Module-level (the owner's actual goal) is additional
work on top of that.
