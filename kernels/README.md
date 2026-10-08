# Kernel development and comparisons

Production native CUDA/C++ and its adapters belong to `ddo-vmm-allocator`.
This directory is reserved for optional, isolated kernel comparison tooling;
it is not part of the installed DDO package or a runtime dependency.

`benchmarks/external_qk_rope/` contains a build-only adapter for a separately
identified external comparison. Retain external license and source provenance
in any local checkout; do not present third-party kernels as independent DDO
implementations. Keep downloads and generated binaries ignored. Model-specific
runner integration stays in the host repository. Any integration into shipped
code requires a separate explicit decision and attribution review.
