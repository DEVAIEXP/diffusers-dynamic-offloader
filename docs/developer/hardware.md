# Test hardware and environments

This is a reference test host, not DDO's minimum hardware requirement.
Capacity and timing depend on the model, route, precision and component strategy.

| Component | Reference configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Dedicated GPU memory | 8 GiB (8192 MiB) |
| CPU | AMD Ryzen 5 5600X, 6 cores / 12 logical processors |
| Installed system RAM | 64 GiB |

CPU and installed RAM were confirmed on the Windows host on 2026-10-15.
That inspection does not reconstruct every earlier test environment.

| Target | Recorded environment |
| --- | --- |
| Windows | Windows 10 Enterprise build 19045 in the host inspection; exact OS/driver versions were not recorded uniformly for every workload |
| Native Linux | Ubuntu 26.04.1 x86_64; selected installed-wheel validation records driver 595.99.02 |
| WSL2 | Ubuntu 22.04.3; kernel 6.18.33.2-microsoft-standard-WSL2; approximately 39 GiB guest RAM and 32 GiB configured swap; Windows GPU driver |

Configured swap is not evidence that a run used it. WSL guest RAM is not the
host's full installed RAM. Storage model, interface and throughput were not
recorded consistently; storage equivalence is unavailable.

Selected installed-wheel checks use Python 3.12 and Torch 2.12.1+cu132.
These are validation conditions, not a claim that every dependency combination
allowed by package metadata was tested. See [validation scope](validation.md).

The [examples hardware reference](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/hardware.md)
and model benchmark pages own workload resolutions, steps, conditioning,
component routes and measured time/memory. Match those conditions before
comparing results. Record your own hardware, driver, software versions,
storage and VM/container limits rather than assuming this host's values.
