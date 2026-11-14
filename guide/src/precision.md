# Precision

This chapter is not written yet. `f32` is the default; [`Df32`](https://docs.rs/luma/latest/luma/struct.Df32.html) is a double-float of about 48 mantissa bits that runs on every device, `Array<f64>` needs `SHADER_F64`, and `to_f64` / `to_df32` / `to_f32` convert between them. `sum_compensated` and `norm_compensated` accumulate in double-float, `is_finite` / `is_nan` / `is_inf` give lazy masks, and [`ContextOptions::fast_math`](https://docs.rs/luma/latest/luma/struct.ContextOptions.html) allows value-changing rewrites (see [Environment variables](environment.md)).
