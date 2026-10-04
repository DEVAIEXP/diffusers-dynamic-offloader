# Manual validation before release

Other models/platform environments receive user-driven validation; no exhaustive
matrix or automatic release is implied. Keep the accepted benchmark intact and
use a new output directory for each manual case. Do not overwrite historical
JSON/images or alter driver/kernel policy to force a run.

## Before running

- Use DDO's Windows `.venv-vmm` or Linux `.venv-linux`, with the local editable
  DDO and allocator installed. Preserve separate Windows/Linux builds.
- Record model/checkpoint revision, runner revision, DDO/allocator commits,
  library SHA256, Torch/Diffusers versions, driver, GPU and available RAM.
- Select the intended route explicitly. Native VMM requires the first-action
  fresh bootstrap and matching platform binary; linear defaults stay unchanged.
- Use a runner appropriate for the model. The LTX harness is not a universal
  UNet/model runner. A model-agnostic allocator does not imply every loader,
  tensor layout, custom forward or component is validated.
- Keep input/precision/guidance/settings fixed when comparing runtimes. Record
  any change; do not call different workloads a matched benchmark.
- Inspect intended vs actually executed component routes in public reports.
  Encoder component ownership is explicit opt-in, not inferred support.

## What to save

Save the normal image/output and DDO JSON report through the host's tracker or
`metrics_snapshot().as_dict()`. Capture the complete error traceback if a run
fails. The [metrics catalog](metrics.md) explains diagnostic fields.

Check:

1. Output correctness/plausibility and actual route; missing fields are not zero.
2. Setup, first step, steady state and whole-pipeline timings separately;
   CUDA events vs host callbacks must retain their distinct labels.
3. GPU baseline-relative delta vs sampled absolute driver usage, and RAM RSS.
4. Unexpected fallback, registration/admission failure, source read failure,
   quarantine, execution mismatch or outstanding owners at completion.
5. Normal component unload between phases; any exception/cancellation behavior
   is a distinct robustness observation, not proof from a successful image.

Do not infer driver-loss recovery or simultaneous native components from a
single completed generation. Stop on memory anomalies/errors and retain the
report; do not auto-increase budgets, reset the device or repeat indefinitely.
For development automation, new CUDA cases still require prior approval.

## Reporting a problem

Provide the recorded environment/versions, exact command, executed routes,
JSON and error traceback, plus whether it was the first or a reused generation.
If useful, compare the same case using an existing linear/compatibility route,
without deleting native evidence. Publication/push remains the user's decision.
