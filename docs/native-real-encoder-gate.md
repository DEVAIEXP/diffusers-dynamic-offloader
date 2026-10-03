# Real encoder loaded-source preflight (PASS, 2026-10-03)

The exact approved CPU command below exited successfully. All 1,065 source
groups bound, zero missing/rejected/ineligible groups/aliases, zero invalid or
ambiguous headers. Shared embedding/output identity and BF16 parameter total
24,374,650,080 bytes passed. Aligned model payload 24,374,652,672 bytes; 82 units,
six buffer groups totaling 34,818 bytes. Largest slot payload 2,013,757,440 bytes;
two-slot payload 4,027,514,880 bytes (not native page-rounded residency).

Host elapsed 0.5996301 s excludes interpreter/import startup; post-load RSS
719,892,480 bytes, available RAM before 47,895,617,536 bytes. RSS is a single
sample, not peak or all payload resident/read. Safetensor-backed CPU tensors
may map lazy pages; descriptor binding does not validate every payload byte.
CUDA remained uninitialized, no native attach or numerical inference attempted.
JSON: sibling outputs/text_encoder_vmm_gate/loaded_source_preflight_20261003.json.
No retry, DLL promotion, budget/default change or further CUDA action occurred.

The seven-case small component CUDA prerequisite passed in 5.78 s. This gate
does not change the protected transformer or host encoder default. Host-specific
loader code remains in the sibling repository; generic geometry/source lookup
is owned by DDO and its native adapter.

## First approved scope: one CPU encoder, no native attachment

Hypothesis: the actual BF16 loader's 1,065 unique parameters associate with
immutable source descriptors, preserving the shared embedding/output alias;
generic execution-unit geometry can be observed without CUDA or native allocation.
The exact local checkpoint must match the previous metadata audit:
24,374,650,080 parameter bytes. Buffer ceiling remains 1 MiB.

`DynamicOffloadLoadResult.inspect_vmm_component()` returns a detached public
report with source coverage and component geometry. Slots use the existing
512-byte parameter packing; reported payload is not page-rounded native
allocation, residency, an admission decision or a support/performance claim.
No automatic budget calibration, source-cache admission or CPU copies are added.

The host loads ONE encoder, with local_files_only and low_cpu_mem_usage.
Before loading, available system RAM must exceed model parameter bytes plus
4 GiB headroom. Post-load process RSS must be <=40 GiB. These checks do not
monitor/enforce the transient loader peak. No downloads, tokenizer, prompt,
transformer, VAE, native allocator/bootstrap, CUDA tensors or denoising.
The command uses DDO's venv, regardless of the sibling script location.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
& .\.venv-vmm\Scripts\python.exe E:\ProjetosIA\ltx2.3-image-diffusers\inspect_loaded_text_encoder_vmm.py 'F:\hf_cache\hub\models--elismasilva--ltx2.3-image-distilled-1.1\snapshots\5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9\text_encoder' --expected-parameter-groups 1065 --expected-parameter-bytes 24374650080 --ram-limit-gb 40 --output E:\ProjetosIA\ltx2.3-image-diffusers\outputs\text_encoder_vmm_gate\loaded_source_preflight_20261003.json
```

Acceptance: full source coverage and expected parameter bytes; zero missing,
rejected or ineligible source groups/aliases; shared embedding/head identity;
eval/contiguous ordinary CPU tensors and buffer limit accepted; detached unit
geometry emitted to JSON; CUDA remains uninitialized, no attachment attempted.
Stop on any failed precondition. Output must not already exist; no retry,
capacity increase or subsequent CUDA action is included in this approval.

Preparation validation: 154 CPU contracts PASS in 6.26 s; script syntax compiled.
At preparation the actual encoder had not been loaded. The subsequently approved
CPU loader/source gate now passed as recorded above; admission/numerical support
remain unvalidated.

## Subsequent gate planning (historical; numerical execution now passed below)

Use observed slot geometry to propose explicit native GPU/source/staging ceilings
before attach. The largest known shared embedding/head is 2,013,757,440 bytes;
two execution slots must fit within the approved GPU ceiling including native
rounding and other reservations. Payload metadata is not proof of driver admission.
Never increase ceilings implicitly or reuse transformer component capacity records.

Then compare the actual prompt embeddings and attention mask with the existing
group-offload route in sequential fresh processes, never keeping two full
encoders live. Preserve the host's real tokenization, SDPA and forward path.
Define exact inputs, tolerance, memory monitoring and cleanup acceptance before
approval. Only afterward consider the full phase-staged performance benchmark.

## Encoder-only numerical gate (approved and executed: PASS, 2026-10-03)

Exact two-process sequence below exited0. Both native prompt encodes matched
the ordinary group-offload reference within BF16 rtol/atol0.016; attention mask
exact and all outputs finite. All1,065 native file bindings/zero CPU-source
bindings; original loader Parameters released, native parameters CUDA, unload
retired them to meta and preserved shared embedding/head identity.

Observed phase wall times (include CPU readback/output artifact or comparison,
public metrics and sampling; not exclusive GPU forward timing):

| Phase | Group reference (s) | Native (s) |
| --- | ---: | ---: |
| Load, attach, tokenizer/pipeline setup | 2.2579 | 4.7261 |
| First encode | 16.2952 | 6.4763 |
| Second encode | not run | 2.4827 |
| Explicit native unload | not measured | 2.3310 |

The reference unload phase did not destroy its model; process exit releases it.
Its0.0s marker is not comparable to explicit native unload. Imports/bootstrap,
reference artifact reload before encoding and final process exit are not included
in these phase times. No throughput target or full pipeline speedup is claimed.

Source read/committed/registered bytes remained23,555,211,264 across native
calls. Staging registered bytes1,610,612,736. DMA25,545,825,792 then51,091,651,584
cumulative: warm source reuse does not eliminate weight DMA.
Actual native slots round to2,015,363,072 each; total4,030,726,144 bytes, unchanged
and within4GiB. Zero fault failure, temporary fallback, source admission denial,
bypass, allocator quarantine or collected transfer owners; tickets38/47 ->0.
No post-unload global resource gauge was taken, so this is not proof of global
driver-resource leak absence.

Reference peak RSS26.8873GiB, native peak RSS27.7958GiB including unload.
Maximum native GPU delta5.0332GiB above approximately1.03GiB baseline, not total
GPU/shared-WDDM memory. All monitor errors null; sampled limits passed. Candidate
and protected default hashes unchanged. Diffusers reference tracing warns about
visual branches unused for text-only input; reference nevertheless passed.
Artifacts in sibling outputs/text_encoder_vmm_gate: reference_20261003.pt,
reference_20261003.json, native_20261003.json.
No retry, budget change, DLL promotion, host default migration or denoising.
This proves the selected single real prompt at1024 tokens, not arbitrary lengths,
multimodal branches, long-running reuse/cancellation or full staged integration.

Sibling `validate_text_encoder_vmm.py` is a separate opt-in diagnostic, not a
default runner change. Preparation syntax compilation preceded the approved run.
One ordinary allocator process creates the group-offload reference artifact;
only on successful exit does a second process start native bootstrap and compare
two encoder calls. Never two live encoders in one process. DDO Python runs both.

Use the existing real cat prompt (fixed in the harness), seed43, BF16, sequence
length1024, guidance1/no negative encoding, unchanged local modular prompt step
and tokenizer. Local files only and local .hf_modules cache; no download, custom
attention substitution, denoising, transformer, connectors or VAE execution.
Loader and forward remain the host's existing Gemma conventions; native bootstrap
executes before Torch imports and installs the existing lifecycle OFF candidate.

Fixed native limits: GPU weight ceiling4GiB; bootstrap allocator cache512MiB;
aggregate staging1.5GiB; page2MiB; persistent source/registered ceilings24.625GiB;
bounded component buffers1MiB. Two execution-slot payloads3.751GiB are below
4GiB before native rounding/reservations; native admission is still a gate, not
assumed from this calculation. No automatic adjustment. DLL SHA256
755FCDD8DD1A72B371402750C071CA4B2146C0AC6BFE6B94C9B78F50F1EBB729.

Monitoring uses existing RunTracker RSS mode, default0.1s cadence. Available RAM
before load must exceed parameter bytes+4GiB; sampled process RSS must be<=40GiB,
sampled absolute driver GPU use<=7.5GiB. Limits are checked after each phase;
they do not enforce a hard instantaneous cap. This is not shared-WDDM accounting.
Stop at first process failure, failed acceptance or sampled memory anomaly.

```powershell
Set-Location E:\ProjetosIA\diffusers-dynamic-offloader
$env:PYTHONPATH='E:\ProjetosIA\diffusers-dynamic-offloader\src;E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\src'
$env:DDO_VMM_LIBRARY_PATH='E:\ProjetosIA\diffusers-dynamic-offloader\ddo-vmm-allocator\build-lifecycle-off-20261002\Release\ddo_vmm.dll'
& .\.venv-vmm\Scripts\python.exe E:\ProjetosIA\ltx2.3-image-diffusers\validate_text_encoder_vmm.py --model-path 'F:\hf_cache\hub\models--elismasilva--ltx2.3-image-distilled-1.1\snapshots\5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9' --ddo-backend linear --reference E:\ProjetosIA\ltx2.3-image-diffusers\outputs\text_encoder_vmm_gate\reference_20261003.pt --report E:\ProjetosIA\ltx2.3-image-diffusers\outputs\text_encoder_vmm_gate\reference_20261003.json
if ($LASTEXITCODE -ne 0) { throw 'Reference encoder gate failed; stop here' }
& .\.venv-vmm\Scripts\python.exe E:\ProjetosIA\ltx2.3-image-diffusers\validate_text_encoder_vmm.py --model-path 'F:\hf_cache\hub\models--elismasilva--ltx2.3-image-distilled-1.1\snapshots\5cd2f057b18d3964f7dc1ce9ab061e9dd22754b9' --ddo-backend vmm --ddo-preset auto --reference E:\ProjetosIA\ltx2.3-image-diffusers\outputs\text_encoder_vmm_gate\reference_20261003.pt --report E:\ProjetosIA\ltx2.3-image-diffusers\outputs\text_encoder_vmm_gate\native_20261003.json
if ($LASTEXITCODE -ne 0) { throw 'Native encoder gate failed; no retry or further case' }
```

Acceptance: loaded preflight still covers1065 BF16 source groups and expected
bytes; all native parameters CUDA and original loader Parameters released;
both native calls match reference embeddings within rtol/atol0.016, attention
mask exact, no nonfinite output. Zero native CPU-source bindings, fault failure,
temporary fallback, source admission denial/bypass or collected transfer-owner
leak; source reads unchanged between the two calls and both source ceilings
respected. Native unload retires parameters to meta and preserves shared aliases.
Memory monitor must complete without error; successful phase and public DDO
reports saved separately. Timings are diagnostic, not a performance target.
No real-encoder interruption injection, pipeline benchmark, compatibility removal,
DLL promotion, history cleanup, budget increase or automatic retry is included.
