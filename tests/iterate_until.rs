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
fn stops_after_the_blocks_have_grown() {
    // The change of step k is 0.5^k, first at most 1e-16 at k = 54: past the blocks of 16 and 32.
    let x = Array::from_slice(&[1.0f32; 100], 100).unwrap();
    let got = x.iterate_until(200, 1e-16, halve).to_vec().unwrap();
    let mut host = 1.0f32;
    for _ in 0..200 {
        let next = host * 0.5;
        let change = host - next;
        host = next;
        if change <= 1e-16 {
            break;
        }
    }
    assert_eq!(host, 0.5f32.powi(54));
    assert_close(&got, &[host; 100], 1e-6, 0.0);
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
        let change = (&next - &u).abs().max().item().unwrap();
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

#[test]
fn assigning_into_the_state_of_a_step_is_read_only() {
    let x = Array::from_slice(&[1.0f32; 16], 16).unwrap();
    let error = std::cell::RefCell::new(None);
    let _ = x
        .iterate_until(3, 0.0, |u| {
            if let Err(e) = u.slice(0, ..1).assign(0.0) {
                *error.borrow_mut() = Some(e);
            }
            u * 0.5
        })
        .to_vec();
    let error = error.into_inner().expect("the assignment must fail");
    assert!(matches!(error.root(), LumaError::ReadOnly(_)), "{error}");
}
