# Validation history

These RC2/RC3 records describe earlier artifacts. Their hashes, tests and
limitations are retained for provenance; they do not establish RC5 acceptance.
See [Validation](validation.md) for current RC5 scope and
[Package delivery](releases.md) for distribution records.

## Selected installed-wheel acceptance

The RC3 publication record identifies pair `0.1.0rc3`; see its
[final artifact hashes and registry checks](releases.md#published-rc3).
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
