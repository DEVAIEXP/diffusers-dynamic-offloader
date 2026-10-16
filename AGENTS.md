# DDO repository guidance
## Purpose and ownership
DDO is a generic Diffusers-style offload manager, not a model pipeline repository.
Read docs/README.md, docs/developer/architecture.md, support.md, validation.md and
releases.md for current contracts. Public docs must work without archived handoffs.
Model runners/benchmarks belong to sibling ddo-examples; custom blocks belong to
their independent model repositories. Native C++/CUDA and adapters belong to the
pinned ddo-vmm-allocator submodule. Read its AGENTS.md before native changes.
## Language and evidence
Write all new/updated documentation, comments and explanatory docstrings in English.
Conversation may use the user's language. Preserve verbatim logs, prompts, quotes
and required license/attribution text; never alter evidence to translate or sanitize it.
Use docs/user for current integration, docs/developer for contributor contracts,
and ignored .internal/documentation for handoffs/plans/intermediate investigations.
README must index public docs. Never link public instructions to required ignored files.
Keep a separate backup of local internal records; .gitignore is not a backup and
does not remove previously tracked content from Git history. No implicit history rewrite.
## Protected runtime routes
- linear_runtime is the stable PyTorch-backed route for one_shot_fast,
  low_ram_safe, wsl_compat and warm_process. Preserve behavior and dependencies.
- vmm_runtime is the protected NVMe baseline, not the target for new native work.
- vmm_fast_runtime is the experimental native arena/source/pressure route.
- diffusers_offload_compat, diffusers_leaf_offload_compat and off preserve
  compatibility/plan-mode behavior and backend-specific forwards.
linear/auto preserves one_shot_fast; vmm/auto selects vmm_fast;
vmm/nvme_baseline selects the protected fallback. Do not infer a new route from
class names or add model-specific policies. Ordinary installs remain lightweight.
## Native contracts
Bootstrap must run as the runner's first executable action, relaunch the same
command in a fresh child and install the allocator before CUDA-owning imports.
DDO owns settings, loading/source association, hooks, cleanup, fallback and reports.
Hosts parse --ddo-backend/--ddo-preset and orchestrate phases; they must not build
allocator policy from internal environment variables or hook order.
Native C++ owns stable virtual addresses, physical residency, signatures, eviction,
fixed budgets, reusable slots, event fences and transfer accounting. No Python
prefix/suffix pinning heuristics or parallel source cache in runners.
Retain source descriptors/native bytes, not loader CPU tensors. Storage, CUDA
registration and GPU ceilings are separate. Unregister must preserve content.
Slot/page/ticket reuse requires last upload/consumer completion; partial failures
retain ownership. Never force-free after unproven/poisoned CUDA completion.
Linux admission preserves actual overcommit mode, MemAvailable, cgroup v2 and
process limits. Enforce CommitLimit in mode2, finite physical bounds in modes0/1;
unknown observations refuse growth. WSL and native Linux are separate targets.
## Components, quantization and reporting
Keep VAE routing in enable_offload: resolved auto/one_shot_fast/low_ram_safe/
vmm_fast select no-stream leaf; warm_process/wsl_compat/nvme_baseline preserve
resident staged-host behavior. Explicit routes/configuration remain authoritative.
Normal-allocator dense convolution is opt-in. Native VAE requires explicit
source/capacity setup; routing does not authorize cache trim or cuBLAS clearing.
Saved TorchAO FP8 weight-only payload loading/transport is model-agnostic;
backend math remains TorchAO/Diffusers. Compilation is optional and separate.
Preserve SDNQ/backend forwards and compatibility routes; no universal quantization
recipe acceptance. Timestep/residual/RoPE model math stays outside DDO.
Hosts consume DynamicOffloadMetrics, as_dict and format_profile, not state/
planner_decisions/native internal attributes. Profiles observe component capacity,
not Python residency calibration. GPU deltas and absolute driver peaks are distinct.
Windows standby purge remains Windows-only and safe/no-op elsewhere.
## Documentation, tests and packages
Pipeline results are owned by ddo-examples, with scope/reproduction by model.
Current native selected acceptance is summarized publicly in developer validation;
full original gates/handoffs are local archives, not authorization to rerun.
Use separate platform environments and new candidate build paths. Never overwrite
accepted build-linux/Windows artifacts. Optional vmm-dev/vmm-dev-linux profiles
are maintainer tools, not ordinary requirements.
Wheels must identify artifact hash, API/ABI, build features and compatibility.
No import-time/inference build or download. Source tests do not grant clean-wheel
acceptance. Commit allocator first, update the DDO gitlink second, host separately
only when authorized; ignored model clones require their own commits.
## Execution, validation and release authority
Read this file and relevant public docs before acting. Inspect git status and preserve
user changes. Use apply_patch for manual edits; scope searches/edits to the task.
Before CUDA, benchmarks, benchmark parameter changes, compatibility removal or
history cleanup, show exact command, hypothesis, fixed limits and criteria and
wait for user approval. CPU tests/builds do not authorize GPU execution.
Stop at the first failure/anomaly. No silent retries, budget growth, driver resets,
OS memory-policy changes, binary promotion or full Cartesian test matrix.
Never claim general support from selected acceptance. Treat missing fields as
unavailable, not zero; preserve raw result identity and rejected experiments.
Experimental prereleases precede stable release. Tags, push, Hub/registry upload,
CI publication and Git-history rewrites require explicit permission.
Do not remove required licenses/attribution or suggest that hiding names establishes
independent implementation/license compatibility. Do not commit user scratch files,
weights, environments, build products, outputs or internal records.
