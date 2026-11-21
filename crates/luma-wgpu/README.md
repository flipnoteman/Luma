# luma-hal-wgpu

The [wgpu](https://wgpu.rs) backend of Luma (Vulkan, Metal, DX12). It generates WGSL from the kernel IR and batches work into single compute passes.

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
