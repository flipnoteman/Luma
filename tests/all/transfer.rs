//! Host <-> device transfers of every size class: staged in pieces, through reused staging memory,
//! behind running kernels.

use luma::*;

/// Distinct, position-dependent words: a piece copied to the wrong place or twice shows.
fn pattern(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| i.wrapping_mul(2_654_435_761).rotate_left(i % 29) ^ salt).collect()
}

const MI: usize = 1 << 20;

/// Lengths (in words) around the sizes the backends split at: a few bytes, a page, 1, 2 and 8 MiB
/// pieces, several pieces and a remainder.
fn lengths() -> Vec<usize> {
    vec![1, 3, 1024, 1025, MI / 4 - 1, MI / 4, MI / 4 + 1, MI / 2 + 7, MI - 3, 2 * MI + 1, 2 * MI + 3, 3 * MI + 5, 5 * MI + 123, 9 * MI + 1, 10 * MI + 17]
}

#[test]
fn roundtrip_around_every_piece_size() {
    for n in lengths() {
        let data = pattern(n, 0xA5);
        let a = Array::from_slice(&data, n).unwrap();
        assert!(a.to_vec().unwrap() == data, "to_vec of {n} words differs");
        let mut out = vec![0u32; n];
        a.read_into(&mut out).unwrap();
        assert!(out == data, "read_into of {n} words differs");
    }
}

#[test]
fn read_into_an_unaligned_slice_of_every_size() {
    for n in lengths() {
        let data = pattern(n, 7);
        let a = Array::from_slice(&data, n).unwrap();
        let mut host = vec![0xDEAD_BEEFu32; n + 3];
        a.read_into(&mut host[1..=n]).unwrap();
        assert!(host[0] == 0xDEAD_BEEF && host[n + 1] == 0xDEAD_BEEF && host[n + 2] == 0xDEAD_BEEF, "read_into of {n} words wrote outside its slice");
        assert!(host[1..=n] == data[..], "read_into of {n} words differs");
    }
}

#[test]
fn many_uploads_in_a_row_keep_their_own_data() {
    // The staging memory of an upload is reused as soon as its copy is done: no upload may see
    // another one's bytes, and none may be overwritten before the device has read it.
    let n = 5 * MI + 99;
    let arrays: Vec<(u32, Array<u32>)> = (0..24u32).map(|salt| (salt, Array::from_slice(&pattern(n, salt), n).unwrap())).collect();
    for (salt, array) in &arrays {
        assert!(array.to_vec().unwrap() == pattern(n, *salt), "array {salt} differs");
    }
}

#[test]
fn uploads_and_readbacks_alternate_with_kernels() {
    let n = 3 * MI + 11;
    let mut previous: Option<(Array<u32>, Vec<u32>)> = None;
    for round in 0..6u32 {
        let data = pattern(n, round);
        let a = Array::from_slice(&data, n).unwrap();
        // A kernel chain the upload behind it has to wait for, and a readback of the last round's result.
        let doubled = &a + &a;
        let want: Vec<u32> = data.iter().map(|v| v.wrapping_add(*v)).collect();
        let b = Array::from_slice(&pattern(n, round + 100), n).unwrap();
        let sum = &doubled + &b;
        if let Some((last, last_want)) = previous.take() {
            assert!(last.to_vec().unwrap() == last_want, "round {round}: the previous result differs");
        }
        let sum_want: Vec<u32> = want.iter().zip(pattern(n, round + 100)).map(|(x, y)| x.wrapping_add(y)).collect();
        previous = Some((sum, sum_want));
    }
    if let Some((last, last_want)) = previous {
        assert!(last.to_vec().unwrap() == last_want, "the last result differs");
    }
}

#[test]
fn a_small_readback_after_a_large_one_and_back() {
    let big = pattern(9 * MI + 5, 3);
    let small = pattern(5, 4);
    let a = Array::from_slice(&big, big.len()).unwrap();
    let b = Array::from_slice(&small, small.len()).unwrap();
    for _ in 0..4 {
        assert!(b.to_vec().unwrap() == small);
        assert!(a.to_vec().unwrap() == big);
    }
    assert_eq!(b.to_vec().unwrap(), small);
}

#[test]
fn threads_transfer_at_once() {
    let threads: Vec<_> = (0..4u32)
        .map(|t| {
            std::thread::spawn(move || {
                for round in 0..5u32 {
                    let n = (t as usize + 1) * MI + 31 * round as usize + 1;
                    let data = pattern(n, t * 100 + round);
                    let a = Array::from_slice(&data, n).unwrap();
                    assert!(a.to_vec().unwrap() == data, "thread {t} round {round} differs");
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}
