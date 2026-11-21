# luma-hal-cpu

The native SIMD CPU backend of Luma. It compiles the kernel IR to machine code with Cranelift and runs it across threads (`LUMA_CPU_THREADS`). Luma uses it when no GPU is present.

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
