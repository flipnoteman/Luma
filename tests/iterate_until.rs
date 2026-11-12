//! `iterate_until` stops at the first step whose largest change is within the tolerance, on the
//! GPU, and matches the same loop written on the host.

mod common;

use common::{assert_close, values};
use luma::*;

fn halve(x: &Array<f32>) -> Array<f32> {
    x * 0.5
}

fn heat_step(u: &Array<f32>) -> Array<f32> {
    let lap = u.shift([1isize, 0]) + u.shift([-1isize, 0]) + u.shift([0isize, 1]) + u.shift([0isize, -1]) - u * 4.0;
    u + lap * 0.2
}

#[test]
fn stops_at_the_first_step_within_tol() {
    let x = Array::from_slice(&[1.0f32; 100], 100).unwrap();
    let got = x.iterate_until(100, 1e-3, halve).to_vec().unwrap();
    // The change of step k is 0.5^k, which first is at most 1e-3 at k = 10.
    assert_close(&got, &[0.5f32.powi(10); 100], 1e-6, 0.0);
}

#[test]
fn stops_in_a_later_block() {
    let x = Array::from_slice(&[1.0f32; 100], 100).unwrap();
    let got = x.iterate_until(100, 1e-6, halve).to_vec().unwrap();
    assert_close(&got, &[0.5f32.powi(20); 100], 1e-6, 0.0);
}

#[test]
fn max_steps_bounds_the_loop() {
    let x = Array::from_slice(&[1.0f32; 100], 100).unwrap();
    let got = x.iterate_until(20, 0.0, halve).to_vec().unwrap();
    assert_close(&got, &[0.5f32.powi(20); 100], 1e-6, 0.0);
}

#[test]
fn a_shift_step_matches_the_host_loop() {
    let data = values(64 * 64, 7, 0.0, 1.0);
    let u0 = Array::from_slice(&data, [64, 64]).unwrap();
    let got = u0.iterate_until(300, 1e-5, heat_step).to_vec().unwrap();

    let mut u = &u0 * 1.0;
    for _ in 0..300 {
        let next = heat_step(&u);
        let change = (&next - &u).abs().max().unwrap();
        u = next;
        if change <= 1e-5 {
            break;
        }
    }
    assert_close(&got, &u.to_vec().unwrap(), 1e-5, 1e-5);
}

#[test]
fn zero_steps_returns_a_copy() {
    let data = values(10, 3, 0.0, 1.0);
    let x = Array::from_slice(&data, 10).unwrap();
    let got = x.iterate_until(0, 1e-3, halve);
    assert_eq!(got.to_vec().unwrap(), data);
    assert_eq!(x.to_vec().unwrap(), data);
}
