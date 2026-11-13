use luma::*;
fn cat(axis: usize, parts: &[Array<u32>]) -> Array<u32> {
    match parts { [one] => one.clone(), [a, b] => concat(axis, [a, b]), _ => { let (a, b) = parts.split_at(parts.len() / 2); concat(axis, [&cat(axis, a), &cat(axis, b)]) } }
}
fn main() -> Result<()> {
    let ctx = Context::new()?;
    let n = 589_824usize;
    for w in [42usize, 168] {
        let arrays: Vec<Array<u32>> = (0..w).map(|i| Array::<u32>::range_on(&ctx, [1, n], 1).unwrap() + (i as u32)).collect();
        let refs: Vec<&Array<u32>> = arrays.iter().collect();
        eval_all(&refs)?; ctx.sync()?;
        for rep in 0..3 {
            let (out, prof) = ctx.profile(|| {
                let out = cat(0, &arrays);
                out.eval().unwrap(); ctx.sync().unwrap(); out
            })?;
            if rep == 2 { println!("w-fastest stack w={w} {:?} gpu={:?}", out.shape(), prof.total()); }
        }
    }
    Ok(())
}
