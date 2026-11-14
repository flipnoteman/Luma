# Performance guide

Tools: [`Context::profile`](lazy.md#looking-at-what-ran-profile) times each kernel and lists where an expression was split (`Profile::splits`, [`SplitReason`](https://docs.rs/luma/latest/luma/enum.SplitReason.html)). When an expression can be recomputed, staged in workgroup memory or evaluated first, a cost model measured per device decides: see [`CostModel`](https://docs.rs/luma/latest/luma/struct.CostModel.html) and `Context::cost_model`. The rest of this chapter is not written yet; the settings that shape fusion are in [Environment variables](environment.md) (`LUMA_FUSION_*`, `LUMA_STAGING`, `LUMA_CALIBRATE`, `LUMA_CACHE`).

## The cost model

`Context::cost_model()` returns the throughputs Luma uses to compare plans (launch cost, memory bandwidth, `f32` rate, the cost of `Df32` and `f64` arithmetic relative to `f32`).
