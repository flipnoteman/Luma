# The Luma Guide

The user guide of Luma, an [mdBook](https://rust-lang.github.io/mdBook/) (pinned to
0.5.4 in `.github/workflows/guide.yml`). Published at <https://flipnoteman.github.io/Luma/>, one
directory per version: `/main/` is the tip of `main`, `/vX.Y.Z/` is the guide as of each release.
The API reference is on [docs.rs](https://docs.rs/luma).

`main` holds only the Markdown source (`guide/src/`), the theme (`guide/theme/`) and the workflow;
the built site lives on the `gh-pages` branch.

## Writing

Every Rust code block in `guide/src/*.md` is compiled and run by `cargo test --doc`: `src/lib.rs`
has a `#[cfg(doctest)]` module that includes each chapter. **A new chapter needs a line there**
(and an entry in `guide/src/SUMMARY.md`). Hide setup with `# ` lines as in rustdoc; use
`text` for blocks that are not Rust, and avoid `no_run` / `ignore` unless the example cannot run
on CI's software device.

```sh
cargo test --release --doc guide     # compile and run the guide's examples
```

## Preview

```sh
cargo install mdbook --version 0.5.4 --locked
mdbook serve guide --open            # http://localhost:3000, rebuilds on save
mdbook build guide                   # writes guide/book/
```

The version switcher in the menu bar only appears on the published site (it reads
`/Luma/versions.json`).

## How versions are published

`.github/workflows/guide.yml` builds the guide and commits it to `gh-pages`:

| Trigger | Result |
| --- | --- |
| push to `main` touching `guide/**`, `src/**` or the workflow | publishes `/main/` |
| a GitHub release is published | builds the release tag, publishes `/<tag>/` (for example `/v0.3.0/`) |
| manual run (Actions -> Guide -> Run workflow) with inputs `ref` and `version` | builds `ref`, publishes it as `version` |

After every run the workflow regenerates `/versions.json` (releases newest first, then `main`;
the version switcher reads it) and `/index.html`, which redirects to the newest release (to
`/main/` while there is none). Existing version directories are replaced only when the same
version is published again and are never deleted. The workflow creates `gh-pages` as an orphan
branch on its first run.

### `v0.3.0` and older tags

`v0.3.0` was tagged before `guide/` existed, so no release run can build it. Publish it once by
hand, building today's guide under that name: Actions -> Guide -> Run workflow with
`ref = main` and `version = v0.3.0`. The same run rebuilds any version later (a typo fix in an old
release's guide: use `ref = <tag or commit>` that contains `guide/`, or `main`).

## One-time GitHub setting

After the first run has created the `gh-pages` branch: Settings -> Pages -> Source:
**Deploy from a branch** -> branch `gh-pages`, folder `/ (root)`. The branch contains a
`.nojekyll` file, so Pages serves the files as they are.
