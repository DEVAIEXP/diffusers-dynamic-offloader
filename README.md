# Diffusers Dynamic Offloader

DDO is a model-agnostic offload manager for low-VRAM Diffusers-style inference.
It provides dense-weight routes and official Diffusers compatibility routes
while preserving quantized backend math.

DDO moves model weights between CPU memory and the GPU as components run. It
manages loading, transfers and resource lifetime; Diffusers and the selected
backend still perform the model computation. Encoder, transformer and VAE phases
can use different strategies, including official Diffusers leaf offloading.

## What native VMM adds

VMM means **CUDA Virtual Memory Management**. The optional native allocator keeps
GPU pointer addresses stable while managing the physical memory behind them.
The native fast route uses a bounded weight arena, reusable execution buffers
and completion events to reuse memory safely and upload weights when needed.
It does not reserve all GPU memory, pin every allocation in local VRAM, compress
activations or change the model's attention/math. Windows may still page GPU
allocations under pressure.

The regular **DDO linear** route uses PyTorch-backed offloading and needs no
native allocator. **DDO native VMM** is a separate optional route. Neither is a
replacement for every Diffusers offloading strategy; choose per workload and
component. See [native integration](docs/user/native-vmm.md) and
[architecture](docs/developer/architecture.md) for the contracts.

## Measured example

LTX 2.3 BF16 video, Windows, RTX 3060 Ti **8 GiB**, 64 GiB system RAM:
1280×704, 5 seconds/121 frames, 8 steps. These existing measurements use a shared
Diffusers leaf text encoder and resident temporal VAE; the method column names
the **transformer** route. GPU peaks are sampled absolute driver usage; RAM peaks
are process RSS. Times are seconds and memory is GiB.

| Transformer method | Denoise (s) | Tracked total (s) | Peak GPU (absolute, GiB) | Peak RAM (RSS, GiB) |
| --- | ---: | ---: | ---: | ---: |
| Diffusers streamed leaf | 392.91 | 452.33 | 4.72 | 41.34 |
| DDO linear (`one_shot_fast`) | 262.22 | 336.44 | 4.72 | 47.87 |
| DDO native VMM (`vmm_fast`) | 207.40 | 277.09 | 5.21 | 32.65 |

This is a selected pipeline comparison, not an allocator-only ablation or a
universal speedup. Tracked total excludes command startup and downloads. Read
the [video phases and conditions](https://devaiexp.github.io/ddo-examples/docs/benchmarks/ltx23-video/#windows-bf16)
and [comparison summary](https://devaiexp.github.io/ddo-examples/docs/benchmarks/summary/)
before choosing a route. In the Windows LTX image comparison, linear has the
shorter denoise while native has the shorter total. Leaf VAE can reduce decoder
GPU peaks at additional latency; component strategies can be combined.

## Getting started

Follow the [installation guide](docs/user/installation.md), then choose
[full-pipeline](docs/user/full-pipeline.md), [staged](docs/user/staged-pipeline.md)
or [Modular Diffusers](docs/user/modular-pipeline.md) integration.

Normal linear use requires no native allocator or compiler. Native VMM is optional:
it uses a matching packaged DLL/.so and bootstrap before CUDA initialization.
Experimental version `0.1.0rc3` is available on PyPI. The installation guide
uses published packages, without requiring a source clone or local wheel files.

```python
from diffusers_dynamic_offloader import enable_offload

result = enable_offload(transformer, preset="auto", component="transformer")
transformer = result.module
```

This snippet selects the default linear route. See
[native VMM integration](docs/user/native-vmm.md) for its separate loading contract.

## Documentation

Read the [documentation website](https://devaiexp.github.io/diffusers-dynamic-offloader/)
or browse the Markdown guides below.

- [User guides](docs/user/README.md): installation, integration, presets and reporting.
- [Developer guides](docs/developer/README.md): architecture, diagnostics, builds and validation.

On the website, use the top **User guides**, **Developer guides** and
**Benchmarks** menus to change pages. The left contents list links to sections
within the current page; short sections may already be visible.

## Examples and benchmarks

Model runners, custom-block integration and measured comparisons live in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
See the [test hardware](docs/developer/hardware.md) and
[support boundaries](docs/developer/support.md) when interpreting results.
Selected workloads do not establish universal speedups or model support.

## License

Apache-2.0; preserve required copyright, license and attribution notices.
