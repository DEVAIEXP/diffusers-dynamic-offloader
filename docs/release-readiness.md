# Native delivery review (2026-10-10)

Development acceptance and public distribution are separate. This review adds
no CUDA execution, binary promotion, dependency installation or history rewrite.

## Preserved development state

- Generic saved FP8 adapter: allocator commit `3670f8d`; DDO integration
  `42ed5bf`; optional compute compilation is separate in `3dff1f1`.
- Documentation/evidence checkpoint: DDO `88edcf7`; host responsibility update
  `d227914`. The parent pins the allocator revision. Commits are local, not a
  claim that those revisions are available from the public remotes.
- Selected Windows DLL SHA256 remains
  `EC5AB4700EE5D171CC06D8E9EC99C549F8D6CC3DFBE7FEB2683B93ADD0E829E6`.
  Latest review: 118 selected CPU contracts passed in 14.19 s; no rebuild.
- Selected BF16 Windows/Linux acceptance and newer Windows FP8 acceptance
  retain their distinct workload/binary boundaries. Linux FP8 remains unvalidated.
- Linear and NVMe defaults are preserved. LTX-specific compute changes stay
  in the host; generic FP8 transport does not require those changes.

## Concrete distribution work

1. **Installed-package bootstrap (CPU correction complete):** the review
   reproduced refusal without a checkout even with an installed adapter and
   explicit library. Discovery now accepts the installed package without
   executing its initializer and locates a platform-packaged binary automatically.
   Conventional checkout builds retain precedence; explicit overrides retain
   precedence over both. It never scans experimental builds or builds/downloads
   a binary at execution time. CPU relaunch/discovery contracts cover both
   platforms; no new CUDA acceptance follows. Real clean-wheel integration
   remains pending the binary-delivery decision below.
   Validation: 20 CPU tests passed in 0.22 s, including a fresh interpreter
   locating a deliberately non-importable package without importing Torch/CUDA.
   Neither the Windows DLL nor the Linux .so changed.
2. **Native artifact selection (local Windows assembly complete):** the user
   approved an optional allocator wheel containing the unchanged selected DLL.
   It is tagged `py3-none-win_amd64`, with verified bytes and an artifact manifest;
   ordinary DDO and adapter wheels remain pure. See the
   [artifact/assembly record](native-wheel-delivery.md). Linux wheel assembly
   and publication are not complete.
   Record source revision, build options, dependency environment and hash for
   each selected artifact. A default CMake build is not automatically the
   accepted joined-span build. Do not copy/promote or rebuild binaries silently.
3. **Clean installation (Windows CPU smoke complete):** both local wheels
   installed into a fresh dependency-free environment. Isolated `-I` discovery
   and mocked child launch passed without checkout imports or CUDA dependencies;
   license, tags, binary manifest and RECORD checks passed. This is not CUDA/model
   wheel execution or sdist-install validation. Native adapter and
   TorchAO are not mandatory ordinary DDO dependencies; TorchAO 0.18.0 and the
   tested Diffusers serialization stack are required for the selected FP8 recipe.
   Do not claim latest development gates from an untested clean environment.
4. **Publication review:** audit active deliverable files, engineering records,
   private history and remote reachability before publishing. Preserve truthful
   licensing, attribution and raw acceptance evidence. Any history rewrite,
   compatibility removal or publication requires separate explicit approval.
   Removing provenance references cannot establish license compatibility.

The allocator README also contains historical core-only/future-adapter wording;
its standalone installation/support overview needs reconciliation with current
native transport before release. Parent-relative evidence links assume a DDO
checkout and need a deliberate standalone-documentation destination.

This is a finite delivery checklist, not authorization for another performance
search or an open-ended model/platform test matrix. See
[native support](native-support-status.md) and [generic FP8](fp8-weight-only.md)
for actual accepted scope.
