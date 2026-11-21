# luma-hal-cuda

The native CUDA backend of Luma. It generates NVVM IR from the kernel IR, compiles it with the NVIDIA driver's NVVM (linking NVIDIA's libdevice for math functions), runs work on one ordered stream and replays CUDA graphs.

## Usage

This is an internal crate of [Luma](https://crates.io/crates/luma) and is not meant to be used directly.
Depend on `luma` instead:

```toml
[dependencies]
luma = "0.5"
```

- User guide: <https://flipnoteman.github.io/Luma/>
- API reference: <https://docs.rs/luma>
- Source: <https://github.com/flipnoteman/Luma>

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

The embedded `vendor/libdevice.10.bc` (NVIDIA's libdevice) is not covered by that license: it is
redistributed under the NVIDIA CUDA Toolkit End User License Agreement,
[`vendor/NVIDIA-CUDA-EULA.txt`](vendor/NVIDIA-CUDA-EULA.txt). See [`vendor/README.md`](vendor/README.md).
