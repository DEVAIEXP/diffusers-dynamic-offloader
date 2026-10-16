# Native maintainer builds
Builds belong to the allocator repository. Read its README and developer build guide.
DDO's optional `vmm-dev` (Windows) and `vmm-dev-linux` profiles provide development
dependencies; ordinary library users must not acquire compilers/toolkits automatically.
Use separate environments: Windows `.venv-vmm`, native Linux `.venv-linux`, and
WSL its reviewed environment. Examples use their own platform host environments.
Select CMake features explicitly. Current validated Windows and Linux candidates
do not necessarily have identical feature sets. Never assume CMake defaults
produce the accepted joined-span DLL or the accepted Linux span-pool .so.
Use a new candidate build directory and record revision/dirty state, compiler,
toolkit, options, architecture, dependencies and SHA256. Existing build helpers
have fixed output paths: review them before running, never overwrite a checkpoint.
Do not retag a host-built .so as manylinux without ABI/dependency verification.
For repeatable local prerelease rebuilds, use the isolated locked
[candidate generator](releases.md#one-command-native-linux-candidate-generation).
It creates new source snapshots/build paths and never runs CUDA or publishes.
WSL requires its own compatible build and independent validation; do not install
a Linux NVIDIA kernel driver inside WSL. Builds do not run/authorize CUDA tests.
