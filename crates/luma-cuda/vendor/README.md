# Third-party files

`libdevice.10.bc` is NVIDIA's libdevice math library (LLVM bitcode) from CUDA 12.0.140
(`cuda-nvcc-12-0-12.0.140`, sha256 `3c8055bc044e31ac849dadeaac3f4e9dd1d68fd349f647bbf9c507a27ff6ae9e`).
It is not covered by luma's MIT/Apache-2.0 licence: it is redistributed under the NVIDIA CUDA
Toolkit End User License Agreement, which lists it as redistributable. The agreement is in
[`NVIDIA-CUDA-EULA.txt`](NVIDIA-CUDA-EULA.txt).

`luma-cuda` embeds it (`include_bytes!`) and links it into every kernel through the NVIDIA driver's
NVVM library, for the math functions (`sin`, `exp`, `pow`, ...).
