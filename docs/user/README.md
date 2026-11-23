# User guides

DDO manages weight loading, transfers and cleanup while Diffusers or the selected
backend computes the model. Choose between two families:

- **Linear:** PyTorch-backed offloading, with no native allocator requirement.
- **VMM:** optional native memory management that keeps tensor GPU addresses
  stable while reusing physical backing and delivering weights as needed.

Start with [runtimes and presets](presets-and-platforms.md), then
[installation](installation.md) and the common [integration guide](integration.md).
The same component APIs apply to both families; VMM Fast adds bootstrap and
source-aware loading before inference.

## Integrate a pipeline

- [Full pipeline](full-pipeline.md): let the pipeline orchestrate its components.
- [Staged pipeline](staged-pipeline.md): load and retire components phase by phase.
- [Modular Diffusers](modular-pipeline.md): construct the blocks needed by each phase.
- [Dynamic loading](dynamic-loading.md): associate supported checkpoint sources
  before attaching offload.

## Choose and measure a configuration

- [Component strategies](component-strategies.md): encoder, transformer and VAE routes.
- [Capacity profiles](capacity-profiles.md): build and reuse workload budgets.
- [Configuration](configuration.md): Python fields, environment overrides and lifecycle options.
- [Metrics](metrics.md): public reports, collection scope and optional diagnostics.
- [Errors and troubleshooting](troubleshooting.md): GPU memory errors, recovery messages and bootstrap errors.

## Optional capabilities

- [Saved FP8 weight-only](fp8-weight-only.md)
- [FP8 conversion](fp8-weight-only-conversion.md)
- [Dense convolution offload](linear-convolution-offload.md)

For ready-to-run model recipes and measured comparisons, use
[DDO examples](https://devaiexp.github.io/ddo-examples/).
For source development and native contracts, use [Developer Guides](../developer/README.md).
