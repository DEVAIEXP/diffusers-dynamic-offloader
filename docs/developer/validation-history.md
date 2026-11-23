# Validation history

These RC2/RC3 records describe earlier artifacts. Their hashes, tests and
limitations are retained for provenance; they do not establish RC5 acceptance.
See [Validation](validation.md) for current RC5 scope and
[Package delivery](releases.md) for distribution records.

## Published rc3

Version `0.1.0rc3` was published on 2026-10-30 for
[DDO](https://pypi.org/project/diffusers-dynamic-offloader/0.1.0rc3/) and the
[allocator](https://pypi.org/project/ddo-vmm-allocator/0.1.0rc3/).

The local `0.1.0rc3` Linux wheel pair has selected installed-wheel image
functional acceptance on native Linux and WSL using the same packaged bytes.
See [historical validation and artifact identities](validation-history.md#local-rc3-linux-wheel-pair).
The candidate contains shared execution-slot release/recreation support;
automatic pressure-triggered reclaim is Windows-only, opt-in and separately
validated. Activation NVMe eviction and compression experiments are not part of
this release scope. Detailed operator, ETW and memory-access investigations are
not enabled by ordinary report collection.

The published files are the final distribution assembled on
2026-10-30 passed auditwheel 6.4.2 repair for `manylinux_2_35_x86_64`, excluding
the external system `libcuda.so.1`. Repair preserved every native library byte;
the NVIDIA driver is not bundled. The earlier builder output retains its
original `linux_x86_64` tag and identity.

| Published wheel | SHA256 |
| --- | --- |
| `diffusers_dynamic_offloader-0.1.0rc3-py3-none-any.whl` | `d9cc131b40459c251a667b58b748675856325b3da7e66335f35b5f2adcb8898e` |
| `ddo_vmm_allocator-0.1.0rc3-py3-none-win_amd64.whl` | `45e361c6ea2706bd810ce408306e889e2b8465be4aafba8d8e56f6b284e86644` |
| `ddo_vmm_allocator-0.1.0rc3-py3-none-manylinux_2_35_x86_64.whl` | `018fb09d2b95d07bd5987702172252bcfc1febb7be609a0725d684322eef5364` |

All three files passed strict metadata validation with twine 7.0.0 and
packaging 26.3. The final Windows and Linux pairs passed isolated CPU metadata,
hash, discovery and mocked bootstrap-child checks. These checks do not load
CUDA. The selected model validations used the same native payloads; the
final repaired Linux archive has not itself undergone a new model run.

The source snapshots were matched to allocator commit `1528358` and DDO commit
`7841e54`, including the allocator gitlink. Registry metadata and downloaded
files matched all three hashes above. A clean Windows installation from PyPI
passed dependency-free discovery and simulated bootstrap checks without CUDA;
this does not replace the selected platform/model validation. No Git tags or
pushes were performed as part of publication. Rebuilding changes artifact
identity and must not inherit the previous wheel's acceptance automatically.

## Selected installed-wheel acceptance

The RC3 publication record identifies pair `0.1.0rc3`; see its
[final artifact hashes and registry checks](#published-rc3).
The final distribution passed strict metadata validation, clean Windows and
Ubuntu 22.04 container CPU installation checks, and a clean Windows PyPI
installation/discovery check without CUDA. Model validation below identifies
the original wheel containers and native payloads separately; registry
publication does not broaden their acceptance scope.

The `0.1.0rc2` allocator/DDO candidate pair has selected installed allocator
smoke-test acceptance on Windows, native Linux and WSL. The unchanged Linux
wheel was used on both Linux targets.

| Target | Native artifact SHA256 | Scope |
| --- | --- | --- |
| Windows AMD64, CUDA 13.2 | `4354a4dce319559c3cbb3533554798d688e9b779f2ca08991f30e28309e69c0b` | Installed discovery and idle-pool trim |
| Linux x86_64 / WSL2, CUDA 13.2 | `ea8f8cc61b2303c4921f1527eae53250833613c83db48b5d2111f1d79309ab59` | Installed discovery, dependency resolution and idle-pool trim |

The reproducible contract is
`ddo-vmm-allocator/scripts/vmm_idle_trim_probe.py --library <installed-library>`.
It preserves a live payload while reclaiming idle backing, permits allocation
after trim and checks final ownership/failure counters. It is not a model or
source-cache performance test.

[Hardware and environments](hardware.md) records the selected test host.
[Package delivery](releases.md) describes candidate generation and identity.
These results do not establish dependency-complete model integration, arbitrary
driver support or manylinux compliance.

### Local rc3 Linux wheel pair

The locally assembled `0.1.0rc3` DDO/allocator pair has selected installed-wheel
LTX image functional acceptance on native Linux and WSL2. Both targets used the
same Docker-produced wheels without rebuilding the native library:

| Artifact | SHA256 |
| --- | --- |
| DDO wheel | `7b5554f2b0c9cb6a2863f02d253b7aedd1dbe08ceff74ba5bcceefe4dc9634ec` |
| Linux allocator wheel | `1d82ef5011030aa4052d7b13ad74ed65751652da58a939d534ecf5bddc6b0240` |
| Packaged native ELF | `71183e987bb8b1d6a252c71ab938589ee89ac13f1e48339cb184731569d55535` |

Each target completed its fixed BF16, 1280 x 704, eight-step, seed-43 image
workload including decode, exited 0 and matched its retained rc2 image exactly.
Expected component routes and capacities were checked; available pre-close
native failure/quarantine counters were zero. Final native counters and latent
finiteness were not exposed by the unchanged runner. Returned cleanup and
process exit do not prove final driver-memory release.

The WSL workload used bounded registration; native Linux used full-source
registration. Do not infer equal capacities or compare their speeds as matched
workloads. Native Linux also emitted a preserved group-offload lazy-prefetch
warning for unused vision-tower layers; the text-prompt workload does not cover
image-conditioned execution. See the [platform-specific model conditions](https://github.com/DEVAIEXP/ddo-examples/blob/main/docs/benchmarks/ltx23-image-details.md#installed-rc3-functional-validation).

Separately, the installed Linux library passed the small explicit execution-slot
reclaim/recreation gate on WSL: two 4 MiB buffers, 8 MiB physical/24 MiB virtual
capacity, one 4 MiB release/recreation, stable address and exact payloads, with
leased/pending refusals and checked pool close. This is explicit-reclaim
acceptance; the model runs do not establish automatic pressure-trigger coverage.
Automatic DXGI reclaim remains Windows-only and disabled by default.

These are development artifacts, not a published rc3 or manylinux certification.
Model timings are observations, not performance acceptance. Native Linux, WSL,
Windows, full video and arbitrary models remain separate validation scopes.

## Source-level coverage

### Updated execution-backing candidate

The subsequent local rc3 Windows allocator wheel has selected installed Qwen
image acceptance with opt-in execution-backing reclaim. Its native SHA256 is
`eb0e3d8241817853a724cca11c294c3d98540b20e0285e36d0c0640cb29c995b`;
wheel SHA256 is
`45e361c6ea2706bd810ce408306e889e2b8465be4aafba8d8e56f6b284e86644`.
The fixed BF16, 1024-square, 20-step workload matched its retained image exactly,
completed source-ticket collection and recorded 29 cumulative transformer
pressure queries with zero failures or unavailable observations. No reclaim
trigger occurred; this model result establishes pressure-path reachability,
not release under pressure. A separate small native gate verified idle-only
release/recreation, refusal of leased/pending buffers and exact payloads.

A new Docker-produced Linux wheel pair also passed selected installed LTX image
checks separately on WSL and native Linux, using the same bytes:

| Artifact | SHA256 |
| --- | --- |
| DDO wheel | `b80068a72b8bbb31a0ff1a6700b3958258bc05fc9bb5ef3c22f7357deec0e482` |
| Linux allocator wheel | `125865e1cb9de65e9249ddd12cafbe4086eb28b53d86e83d7898768e112324a2` |
| Native ELF | `3ac4cacb966dacc8ee7e96bf6502d526173e9d8497654272d2bd60f4c9044289` |

Both targets completed the eight-step BF16 1280 x 704 workload, matched their
retained reference pixels and reported zero available pre-close native failure
and quarantine counters. Original platform-specific staging/registration limits
were preserved. Final native counters and latent finiteness remain unavailable;
process exit and returned cleanup do not prove final driver-memory release.
The native Linux text-prompt run retained the warning about unexecuted
vision-tower layers; image-conditioned execution remains unvalidated.

The reuse-pressure hook is Windows/DXGI-only and compiled out on Linux/WSL.
These are functional development-wheel results, not matched performance,
automatic Linux pressure-policy acceptance, manylinux certification or publication.
