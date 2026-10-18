# Diffusers Dynamic Offloader

DDO is a model-agnostic offload manager for low-VRAM Diffusers-style inference.
It provides dense-weight routes and official Diffusers compatibility routes
while preserving quantized backend math.

## Getting started

Follow the [installation guide](docs/user/installation.md), then choose
[full-pipeline](docs/user/full-pipeline.md), [staged](docs/user/staged-pipeline.md)
or [Modular Diffusers](docs/user/modular-pipeline.md) integration.

Normal linear use requires no native allocator or compiler. Native VMM is optional:
it uses a matching packaged DLL/.so and bootstrap before CUDA initialization.
Experimental version `0.1.0rc2` is available on PyPI. The installation guide
uses published packages, without requiring a source clone or local wheel files.

```python
from diffusers_dynamic_offloader import enable_offload

result = enable_offload(transformer, preset="auto", component="transformer")
transformer = result.module
```

This snippet selects the default linear route. See
[native VMM integration](docs/user/native-vmm.md) for its separate loading contract.

## Documentation

- [User guides](docs/user/README.md): installation, integration, presets and reporting.
- [Developer guides](docs/developer/README.md): architecture, diagnostics, builds and validation.

## Examples and benchmarks

Model runners, custom-block integration and measured comparisons live in
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
See the [test hardware](docs/developer/hardware.md) and
[support boundaries](docs/developer/support.md) when interpreting results.
Selected workloads do not establish universal speedups or model support.

## License

Apache-2.0; preserve required copyright, license and attribution notices.
