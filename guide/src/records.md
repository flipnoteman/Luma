# Records

This chapter is not written yet. A record is a struct of arrays declared with [`luma::fields!`](https://docs.rs/luma/latest/luma/macro.fields.html): arithmetic, `select`, views, `assign` and `+=` act on every field in one kernel, and a record can be the state of a loop. See the [`Fields`](https://docs.rs/luma/latest/luma/trait.Fields.html) trait and [`tests/records.rs`](https://github.com/flipnoteman/Luma/blob/main/tests/records.rs).
