//! The environment variables that filter or size the line-solve and factor strategies (`LUMA_TRIDIAGONAL`, `LUMA_NATIVE_SOLVE`,
//! `LUMA_QR_TSQR`, `LUMA_TRSV`, `LUMA_SVD_PRECOND`, `LUMA_CHOL_NB`): each changes the choice as it did before the registry. The
//! environment is set before a child process starts (changing it while tests run is not safe); the child runs one operation on the CPU
//! backend and prints what the decision log chose.

use std::process::Command;

use luma::*;

const CHILD: &str = "factor_env::operation_child";

fn labels(decisions: &[Decision], site: DecisionSite) -> Vec<String> {
    decisions.iter().filter(|d| d.site == site).map(|d| d.chosen.to_string()).collect()
}

/// Runs in a child process of the tests below: `LUMA_FACTOR_ENV_OP` names the operation, the environment is the parent's.
#[test]
#[ignore = "run by the factor_env tests in a child process with their own environment"]
fn operation_child() {
    let op = std::env::var("LUMA_FACTOR_ENV_OP").expect("set by the parent");
    let ctx = Context::with_options(ContextOptions::from_env().calibrate(false)).unwrap();
    ctx.record_decisions(true);
    let started = ctx.profile(|| {
        match op.as_str() {
            "qr" => {
                let a = Array::<f32>::from_slice_on(&ctx, &(0..512 * 16).map(|i| ((i * 7919) % 1000) as f32 * 1e-3).collect::<Vec<_>>(), [16, 512]).unwrap();
                a.qr().qr.eval().unwrap();
            }
            "triangular" => {
                let l = Array::<f32>::from_slice_on(&ctx, &(0..200 * 200).map(|i| if i % 201 == 0 { 4.0 } else { 0.001 }).collect::<Vec<_>>(), [200, 200]).unwrap();
                let b = Array::<f32>::from_slice_on(&ctx, &[1.0; 800], [4, 200]).unwrap();
                l.solve_triangular(&b, Triangle::Lower, false, false).eval().unwrap();
            }
            "svd" => {
                let a = Array::<f32>::from_slice_on(&ctx, &(0..150 * 150).map(|i| if i % 151 == 0 { 3.0 } else { ((i * 37) % 11) as f32 * 0.01 }).collect::<Vec<_>>(), [150, 150]).unwrap();
                a.svd().s.eval().unwrap();
            }
            "cholesky" => {
                let a = Array::<f32>::from_slice_on(&ctx, &(0..300 * 300).map(|i| if i % 301 == 0 { 400.0 } else { 0.01 }).collect::<Vec<_>>(), [300, 300]).unwrap();
                a.cholesky().eval().unwrap();
            }
            "tridiagonal" => {
                let ones = |v: f32| Array::<f32>::full_on(&ctx, [256, 64], v).unwrap();
                solve_tridiagonal(0, &ones(-1.0), &ones(4.0), &ones(-1.0), &ones(1.0)).eval().unwrap();
            }
            other => panic!("unknown operation {other}"),
        }
        ctx.sync().unwrap();
    });
    let launches = started.map(|(_, profile)| profile.launches()).unwrap_or(0);
    let decisions = ctx.take_decisions();
    println!("child-launches={launches}");
    for (site, name) in [(DecisionSite::QrPanel, "qr-panel"), (DecisionSite::FactorPlan, "factor"), (DecisionSite::LineSolvePlan, "line-solve")] {
        for label in labels(&decisions, site) {
            println!("child-{name}={label}");
        }
    }
    for decision in decisions.iter().filter(|d| d.site == DecisionSite::LineSolvePlan) {
        for (name, verdict) in &decision.candidates {
            println!("child-line-solve-candidate={name} {verdict:?}");
        }
    }
}

/// What the child printed for `name` (`child-<name>=...` lines) and its launches.
fn run(op: &str, env: &[(&str, &str)]) -> (Vec<(String, String)>, u32) {
    let exe = std::env::current_exe().expect("the path of this test binary");
    let mut command = Command::new(exe);
    command.args([CHILD, "--exact", "--ignored", "--nocapture", "--test-threads=1"]);
    for var in ["LUMA_BACKEND", "LUMA_TRIDIAGONAL", "LUMA_NATIVE_SOLVE", "LUMA_QR_TSQR", "LUMA_TRSV", "LUMA_SVD_PRECOND", "LUMA_CHOL_NB", "LUMA_STRATEGY"] {
        command.env_remove(var);
    }
    command.env("LUMA_DEVICE", "cpu").env("LUMA_CALIBRATE", "0").env("LUMA_FACTOR_ENV_OP", op);
    for (name, value) in env {
        command.env(name, value);
    }
    let output = command.output().expect("the child process starts");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(output.status.success(), "{op} {env:?} failed:\n{stdout}\n{}", String::from_utf8_lossy(&output.stderr));
    let mut found = Vec::new();
    let mut launches = 0;
    for line in stdout.lines() {
        // libtest prints `test <name> ... ` before the test's own first line.
        let line = line.rsplit_once(" ... ").map_or(line, |(_, rest)| rest);
        if let Some(rest) = line.strip_prefix("child-launches=") {
            launches = rest.parse().unwrap();
        } else if let Some(rest) = line.strip_prefix("child-") {
            let (name, value) = rest.split_once('=').unwrap();
            found.push((name.to_string(), value.to_string()));
        }
    }
    (found, launches)
}

fn has(found: &[(String, String)], name: &str, pred: impl Fn(&str) -> bool) -> bool {
    found.iter().any(|(n, v)| n == name && pred(v))
}

#[test]
fn luma_qr_tsqr_off_takes_the_generic_panel() {
    let (default, _) = run("qr", &[]);
    assert!(has(&default, "qr-panel", |v| v.starts_with("tsqr ")), "{default:?}");
    let (off, _) = run("qr", &[("LUMA_QR_TSQR", "0")]);
    assert!(has(&off, "qr-panel", |v| v == "generic panel"), "{off:?}");
    assert!(!has(&off, "qr-panel", |v| v.starts_with("tsqr ")), "{off:?}");
    assert!(has(&off, "factor", |v| v == "qr_leaf"), "{off:?}");
    assert!(has(&default, "factor", |v| v == "qr_tsqr_panel"), "{default:?}");
}

#[test]
fn luma_trsv_off_takes_the_blocked_solve() {
    let (off, _) = run("triangular", &[("LUMA_TRSV", "0")]);
    assert!(has(&off, "factor", |v| v.starts_with("tri_blocked")), "{off:?}");
    assert!(!has(&off, "factor", |v| v == "tri_trsv"), "{off:?}");
}

#[test]
fn luma_svd_precond_moves_the_qr_route() {
    let (low, _) = run("svd", &[("LUMA_SVD_PRECOND", "100")]);
    assert!(has(&low, "factor", |v| v == "svd_preconditioned"), "{low:?}");
    let (off, _) = run("svd", &[("LUMA_SVD_PRECOND", "off")]);
    assert!(has(&off, "factor", |v| v == "svd_block_jacobi"), "{off:?}");
    let (default, _) = run("svd", &[]);
    assert!(has(&default, "factor", |v| v == "svd_block_jacobi"), "{default:?}");
}

#[test]
fn luma_chol_nb_sizes_the_blocks() {
    let (narrow, _) = run("cholesky", &[("LUMA_CHOL_NB", "32")]);
    let (wide, _) = run("cholesky", &[("LUMA_CHOL_NB", "128")]);
    assert!(has(&narrow, "factor", |v| v == "chol_blocked nb=32"), "{narrow:?}");
    assert!(has(&wide, "factor", |v| v == "chol_blocked nb=128"), "{wide:?}");
}

#[test]
fn luma_tridiagonal_and_native_solve_pick_the_line_solver() {
    let (cyclic, _) = run("tridiagonal", &[("LUMA_TRIDIAGONAL", "cyclic")]);
    assert!(has(&cyclic, "line-solve", |v| v.starts_with("cyclic ")), "{cyclic:?}");
    let (thomas, _) = run("tridiagonal", &[("LUMA_TRIDIAGONAL", "thomas")]);
    assert!(has(&thomas, "line-solve", |v| v.starts_with("thomas ")), "{thomas:?}");
    // The CPU backend solves natively: Thomas on `auto`; with the native solve off the planner is free to take cyclic reduction (few
    // lines of 256 cells), which `thomas` still overrides.
    let (default, _) = run("tridiagonal", &[]);
    assert!(has(&default, "line-solve", |v| v.starts_with("thomas ")), "{default:?}");
    assert!(!has(&default, "line-solve-candidate", |v| v.starts_with("native ")), "the native Thomas solve is the choice: {default:?}");
    let (native_off, _) = run("tridiagonal", &[("LUMA_NATIVE_SOLVE", "off")]);
    assert!(has(&native_off, "line-solve", |v| v.starts_with("thomas ")), "{native_off:?}");
    assert!(has(&native_off, "line-solve-candidate", |v| v.starts_with("native ")), "the native solve is a rejected alternative: {native_off:?}");
}
