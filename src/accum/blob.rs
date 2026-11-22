//! The byte format of serialized states: a magic, a kind, then little-endian integers and `f64`
//! arrays. Every state value is stored as `f64` (a `Df32` is exactly representable), so a state
//! written by an `f64` accumulator is read by a `Df32` one and the other way round, with the
//! precision of the reader.

use crate::error::{LumaError, Result};

pub(crate) const MAGIC: &[u8; 7] = b"LUMAACC";
pub(crate) const VERSION: u8 = 1;

pub(crate) struct Writer(pub Vec<u8>);

impl Writer {
    pub fn new(kind: u8) -> Writer {
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(MAGIC);
        buf.push(VERSION);
        buf.push(kind);
        Writer(buf)
    }

    pub fn u8(&mut self, v: u8) {
        self.0.push(v);
    }

    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    pub fn dims(&mut self, dims: [usize; 4]) {
        for d in dims {
            self.u64(d as u64);
        }
    }

    pub fn f64s(&mut self, values: &[f64]) {
        self.u64(values.len() as u64);
        for &v in values {
            self.f64(v);
        }
    }
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

fn corrupt(what: &str) -> LumaError {
    LumaError::InvalidShape(format!("serialized accumulator state: {what}"))
}

impl<'a> Reader<'a> {
    /// Checks the magic, the version and the kind.
    pub fn new(bytes: &'a [u8], kind: u8) -> Result<Reader<'a>> {
        if bytes.len() < MAGIC.len() + 2 || &bytes[..MAGIC.len()] != MAGIC {
            return Err(corrupt("not a luma accumulator state"));
        }
        if bytes[MAGIC.len()] != VERSION {
            return Err(corrupt("unknown format version"));
        }
        if bytes[MAGIC.len() + 1] != kind {
            return Err(corrupt("the state is of another accumulator type"));
        }
        Ok(Reader { bytes, at: MAGIC.len() + 2 })
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.at.checked_add(n).filter(|&end| end <= self.bytes.len()).ok_or_else(|| corrupt("truncated"))?;
        let out = &self.bytes[self.at..end];
        self.at = end;
        Ok(out)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")))
    }

    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().expect("eight bytes")))
    }

    pub fn dims(&mut self) -> Result<[usize; 4]> {
        let mut out = [0usize; 4];
        for d in &mut out {
            *d = usize::try_from(self.u64()?).map_err(|_| corrupt("dimension"))?;
        }
        Ok(out)
    }

    pub fn f64s(&mut self, expect: usize) -> Result<Vec<f64>> {
        let n = self.u64()? as usize;
        if n != expect {
            return Err(corrupt("array length does not match the shape"));
        }
        let raw = self.take(n.checked_mul(8).ok_or_else(|| corrupt("length"))?)?;
        Ok(raw.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect())
    }

    pub fn finish(self) -> Result<()> {
        if self.at == self.bytes.len() { Ok(()) } else { Err(corrupt("trailing bytes")) }
    }
}
