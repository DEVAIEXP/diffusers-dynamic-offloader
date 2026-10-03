# Generic model reuse and unload/reload gate

Status: approved CUDA sequence executed once: **1 PASS, no skip, 5.25 s**,
exit 0. All 128 outputs matched reference over four A->B->A->B attachments;
source/event ownership stabilized within each attachment, public collection
retired tickets to zero, hooks/registries closed and files were deletable.
Aggregate cache committed/registered bytes were zero before each new first
forward. OFF hash verified before execution; milestone/protected hashes intact.
No runtime patch, benchmark, rerun or promotion; approval exhausted.
Prior fixture/configuration CPU suite: 31 PASS (5.19 s).

Observed source reads per A/B attachment: 8,448 /18,816 B, unchanged from
forward 1 to 32. Source committed/registered storage remained 67,108,864 B.
Transfer events/tickets zero after collection; two live batch completion events
while attached. Free pooled events 2 for first A, 4 for later attachments, stable
within each; do not interpret these different bounded warm values as a leak.
Diagnostic artifact in
`ddo-vmm-allocator/build-lifecycle-off-20261002/pytest-model-lifecycle-gate/test_generic_model_128_forward0/lifecycle_observations.json`.

Hypothesis: the DDO public fast-VMM attachment can reuse a model repeatedly,
remove its hooks and release native source ownership, then load new source
content at the same path without stale descriptors/signatures or accumulated
host registrations. This follows the three passing native source-failure gates.

ONE generic test: four model attachment cycles A->B->A->B, 32 forwards each,
128 total. A/B are two ordinary Linear/SiLU/Linear modules with widths 32/48,
different weights and source sizes. No LTX-specific classes or model downloads.
Each forward compares its CPU-readback output with a CPU reference and takes
the public report plus explicit nonblocking completion collection.

Limits for this new small contract (not a benchmark parameter change): 0.07 GiB
weight ceiling, 2 MiB native staging, one 64 MiB persistent-source window and
64 MiB registration ceiling, 2 MiB arena page. The public source geometry is
64 MiB; no silent rounding from a 2 MiB request. Existing model benchmark
ceilings and all default routes remain unchanged. No fault injection build.

Executed with isolated lifecycle OFF DLL SHA256 (command not approval to rerun)
`755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729`:

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:DDO_VMM_LIBRARY_PATH = 'E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
& .venv-vmm/Scripts/python.exe -m pytest `
  tests/test_vmm_model_lifecycle_cuda.py::test_generic_model_128_forwards_with_unload_reload_and_source_replacement `
  -x -q -p no:cacheprovider `
  --basetemp ddo-vmm-allocator/build-lifecycle-off-20261002/pytest-model-lifecycle-gate
if ($LASTEXITCODE -ne 0) { throw 'Model lifecycle gate failed; stop' }
```

Require ONE PASS, no skip. Stop at first failed assertion/anomaly; no rerun,
extra cycles or ceiling adjustment under this approval.

Criteria:

- All 128 outputs match reference (rtol1e-4/atol1e-5).
- Four file-backed bindings, no retained CPU source bindings, no unexpected
  faults/fallback, execution mismatch, admission denial or source bypass.
- Source bytes, committed/registered bytes, registration count and scoped event
  owner counts stabilize after first forward of each attachment; no further
  source reads. Tickets and transfer events reach zero after public collection.
- Before each new attachment's first forward: aggregate cache committed and
  registered bytes zero, proving prior source cleanup at that boundary. This
  excludes staging and is not total process RAM/GPU accounting.
- Each removal closes the registry/removes module hooks; source file becomes
  deletable. New content/shape at the same path must produce correct outputs.
- Save first/last public reports for each cycle to the pytest case directory as
  `lifecycle_observations.json`. It is diagnostic test output, not runner logic.

Limits of this proof: tiny model without VRAM pressure, public native weight
arena route under ordinary PyTorch allocation, not fresh-child custom-allocator
full inference; explicitly collecting per forward differs from the earlier
no-intermediate-collection generation gate. It does not prove long-duration
driver stability, full-model memory peaks, asynchronous cancellation, LoRA,
quantization or other operator support. Such evidence requires separate gates.
