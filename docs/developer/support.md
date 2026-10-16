# Support and limitations

DDO preserves model/backend computation and manages component loading, offload
and resource lifetime. Support depends on the selected route, payload format
and platform, not only the model class.

| Area | Current boundary |
| --- | --- |
| Linear / compatibility | PyTorch-backed routes and official Diffusers offload; preserve backend forwards |
| Native Windows / native Linux | Selected BF16 and saved TorchAO FP8 workloads; not arbitrary-model certification |
| WSL2 | Selected BF16 workloads; host-registration capacity is constrained and must be measured separately |
| VAE | Preset-based leaf/resident policy; eager convolution and native source/capacity routes have explicit restrictions |
| Saved FP8 | Documented TorchAO weight-only payload contract, not arbitrary FP8 formats |
| SDNQ | Backend-preserving compatibility/residency routes; not native FP8 acceptance |
| macOS / arbitrary drivers / multi-GPU | Not established by selected validation |

The installed-wheel smoke test is narrower than model validation. See
[validation](validation.md) and [test hardware](hardware.md).
Linux and WSL are separate targets even when they use the same wheel.

An 8 GiB GPU result does not establish low-system-RAM support. Host callbacks
are not completed CUDA timings, and pre-unload snapshots do not prove final
driver-memory release. Windows standby purge is optional, Windows-only and
a no-op elsewhere.

Choose routes using [presets](../user/presets-and-platforms.md) and
[component strategies](../user/component-strategies.md). Model timings and
hardware-specific recommendations belong to
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
