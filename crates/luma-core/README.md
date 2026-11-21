# luma-hal-core

The kernel IR and backend interface (`Backend` trait) shared by every Luma backend. Backends lower the kernel IR to their own shading language or machine code and implement `Backend` to allocate buffers, compile kernels and dispatch work.

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
