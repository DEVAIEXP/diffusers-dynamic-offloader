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

The local rc3 Linux wheel pair has selected installed LTX image functional
validation on both targets, with exact reference images. This is development
candidate acceptance, not package-index availability, a general speed guarantee
or full-video coverage. The shared explicit execution-slot reclaim mechanism is
distinct from the Windows-only, default-off automatic DXGI policy. See the
[rc3 validation scope](validation.md#local-rc3-linux-wheel-pair).

An 8 GiB GPU result does not establish low-system-RAM support. Host callbacks
are not completed CUDA timings, and pre-unload snapshots do not prove final
driver-memory release. Windows standby purge is optional, Windows-only and
a no-op elsewhere.

`diagnose_memory_failure(error)` provides a failure-only host diagnostic without
querying CUDA or changing memory policy. On Windows it reports system commit
headroom separately from available physical RAM. Win32 error 1455 identifies a
commit allocation refusal; a generic CUDA fault with low headroom is reported as
correlation, not proof of its cause. The 256 MiB low-headroom label is diagnostic,
not an admission margin. Unavailable observations remain explicit. Runners must
preserve the original exception and must not retry or free unproven CUDA work.
This diagnostic does not prevent exhaustion or establish safe CUDA recovery.

Choose routes using [presets](../user/presets-and-platforms.md) and
[component strategies](../user/component-strategies.md). Model timings and
hardware-specific recommendations belong to
[DDO examples](https://github.com/DEVAIEXP/ddo-examples).
