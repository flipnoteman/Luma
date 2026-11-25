//! `.npy` headers and the `<file>.luma` sidecar.

use std::fmt;

/// Element types of file arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// IEEE half precision.
    F16,
    /// IEEE single precision.
    F32,
    /// IEEE double precision.
    F64,
    /// Unsigned 8-bit.
    U8,
    /// Signed 8-bit.
    I8,
    /// Unsigned 16-bit.
    U16,
    /// Signed 16-bit.
    I16,
    /// Unsigned 32-bit.
    U32,
    /// Signed 32-bit.
    I32,
    /// Complex of two `f32`.
    C64,
}

impl Dtype {
    const ALL: [Dtype; 10] = [
        Dtype::F16,
        Dtype::F32,
        Dtype::F64,
        Dtype::U8,
        Dtype::I8,
        Dtype::U16,
        Dtype::I16,
        Dtype::U32,
        Dtype::I32,
        Dtype::C64,
    ];

    /// Bytes per element.
    pub fn elem_bytes(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8 => 1,
            Dtype::F16 | Dtype::U16 | Dtype::I16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::F64 | Dtype::C64 => 8,
        }
    }

    /// The NumPy `descr` string (`"<f4"`, `"|u1"`, `"<c8"`, ...).
    pub fn npy_descr(self) -> &'static str {
        match self {
            Dtype::F16 => "<f2",
            Dtype::F32 => "<f4",
            Dtype::F64 => "<f8",
            Dtype::U8 => "|u1",
            Dtype::I8 => "|i1",
            Dtype::U16 => "<u2",
            Dtype::I16 => "<i2",
            Dtype::U32 => "<u4",
            Dtype::I32 => "<i4",
            Dtype::C64 => "<c8",
        }
    }

    /// The name used in sidecars (`f32`, `f16`, `c64`, ...).
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F16 => "f16",
            Dtype::F32 => "f32",
            Dtype::F64 => "f64",
            Dtype::U8 => "u8",
            Dtype::I8 => "i8",
            Dtype::U16 => "u16",
            Dtype::I16 => "i16",
            Dtype::U32 => "u32",
            Dtype::I32 => "i32",
            Dtype::C64 => "c64",
        }
    }

    /// The dtype with this sidecar name, if any.
    pub fn from_name(name: &str) -> Option<Dtype> {
        Self::ALL.into_iter().find(|d| d.name() == name)
    }

    /// The dtype of a NumPy `descr` string. One-byte types accept `|` or `<`; everything else
    /// must be little-endian. Big-endian and unsupported types are refused by name.
    pub fn from_npy_descr(descr: &str) -> Result<Dtype, FormatError> {
        if descr.starts_with('>') {
            return Err(FormatError::BigEndian(descr.to_string()));
        }
        let one_byte = match descr {
            "<u1" => Some(Dtype::U8),
            "<i1" => Some(Dtype::I8),
            _ => None,
        };
        one_byte
            .or_else(|| Self::ALL.into_iter().find(|d| d.npy_descr() == descr))
            .ok_or_else(|| FormatError::UnsupportedDtype(descr.to_string()))
    }
}

/// A malformed or unsupported `.npy` header or sidecar. The message names the problem.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// The data does not start with `\x93NUMPY`.
    BadMagic,
    /// The buffer ends before the header does.
    Truncated,
    /// A `.npy` format version other than 1, 2 or 3.
    UnsupportedVersion(u8, u8),
    /// The header is longer than 1 MiB.
    HeaderTooLarge(usize),
    /// The header text does not match the expected dictionary grammar.
    BadHeader(String),
    /// `descr` is a list (a structured dtype).
    StructuredDescr,
    /// A big-endian `descr`.
    BigEndian(String),
    /// A dtype that is not in [`Dtype`] (`|O`, `|b1`, `<M8[ns]`, ...).
    UnsupportedDtype(String),
    /// More than four dimensions.
    RankTooLarge(usize),
    /// A sidecar problem.
    BadSidecar(String),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::BadMagic => write!(f, "not a .npy file: the magic string \\x93NUMPY is missing"),
            FormatError::Truncated => write!(f, ".npy header is truncated"),
            FormatError::UnsupportedVersion(a, b) => write!(f, "unsupported .npy format version {a}.{b}"),
            FormatError::HeaderTooLarge(n) => write!(f, ".npy header of {n} bytes exceeds the 1 MiB limit"),
            FormatError::BadHeader(m) => write!(f, "malformed .npy header: {m}"),
            FormatError::StructuredDescr => write!(f, "structured dtypes (descr is a list) are not supported"),
            FormatError::BigEndian(d) => write!(f, "big-endian dtype {d:?} is not supported"),
            FormatError::UnsupportedDtype(d) => write!(f, "unsupported dtype {d:?}"),
            FormatError::RankTooLarge(r) => write!(f, "rank {r} exceeds the maximum of 4"),
            FormatError::BadSidecar(m) => write!(f, "malformed sidecar: {m}"),
        }
    }
}

impl std::error::Error for FormatError {}

/// A parsed `.npy` header. `shape` is in Luma dims (first dim fastest), see [`parse_npy`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NpyHeader {
    /// Element type.
    pub dtype: Dtype,
    /// Luma dims: the file's shape as is for Fortran order, reversed for C order; `()` is `[1]`.
    pub shape: Vec<usize>,
    /// The file's `fortran_order` flag (data is never reordered).
    pub fortran_order: bool,
    /// Byte offset of the data: the total length of the header.
    pub data_offset: u64,
}

const MAGIC: &[u8; 6] = b"\x93NUMPY";
const MAX_HEADER: usize = 1 << 20;
const MAX_RANK: usize = 4;

struct Cursor<'a> {
    s: &'a [char],
    i: usize,
}

impl Cursor<'_> {
    fn ws(&mut self) {
        while self.s.get(self.i).is_some_and(|c| c.is_whitespace()) {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, c: char) -> Result<(), FormatError> {
        self.ws();
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(bad(format!("expected {c:?} at position {}", self.i)))
        }
    }

    fn string(&mut self) -> Result<String, FormatError> {
        self.ws();
        let q = match self.peek() {
            Some(q @ ('\'' | '"')) => q,
            _ => return Err(bad(format!("expected a quoted string at position {}", self.i))),
        };
        self.i += 1;
        let start = self.i;
        while self.peek().ok_or_else(|| bad("unterminated string".into()))? != q {
            self.i += 1;
        }
        let out = self.s[start..self.i].iter().collect();
        self.i += 1;
        Ok(out)
    }

    fn word(&mut self) -> String {
        self.ws();
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_') {
            self.i += 1;
        }
        self.s[start..self.i].iter().collect()
    }

    fn shape(&mut self) -> Result<Vec<usize>, FormatError> {
        self.expect('(')?;
        let mut dims = Vec::new();
        loop {
            self.ws();
            if self.peek() == Some(')') {
                self.i += 1;
                return Ok(dims);
            }
            let n = self.word();
            dims.push(n.parse::<usize>().map_err(|_| bad(format!("bad shape entry {n:?}")))?);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(')') => {}
                _ => return Err(bad("expected ',' or ')' in shape".into())),
            }
        }
    }
}

fn bad(m: String) -> FormatError {
    FormatError::BadHeader(m)
}

/// Parses a `.npy` header (format v1, v2 or v3) from the start of `head`, which must contain
/// the whole header (the data may follow).
///
/// Shape mapping: `fortran_order: True` maps to Luma dims as they are (first dim fastest); C
/// order maps to the reversed dims; rank 0 is `[1]`; rank above 4 is refused. The data is never
/// reordered.
pub fn parse_npy(head: &[u8]) -> Result<NpyHeader, FormatError> {
    if head.len() < MAGIC.len() {
        return Err(FormatError::Truncated);
    }
    if &head[..6] != MAGIC {
        return Err(FormatError::BadMagic);
    }
    if head.len() < 8 {
        return Err(FormatError::Truncated);
    }
    let (major, minor) = (head[6], head[7]);
    let (prefix, hlen) = match major {
        1 => {
            let b = head.get(8..10).ok_or(FormatError::Truncated)?;
            (10, u16::from_le_bytes([b[0], b[1]]) as usize)
        }
        2 | 3 => {
            let b = head.get(8..12).ok_or(FormatError::Truncated)?;
            (12, u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
        }
        _ => return Err(FormatError::UnsupportedVersion(major, minor)),
    };
    if hlen > MAX_HEADER {
        return Err(FormatError::HeaderTooLarge(hlen));
    }
    let end = prefix + hlen;
    let raw = head.get(prefix..end).ok_or(FormatError::Truncated)?;
    let text: Vec<char> = if major == 3 {
        std::str::from_utf8(raw)
            .map_err(|_| bad("v3 header is not valid UTF-8".into()))?
            .chars()
            .collect()
    } else {
        raw.iter().map(|&b| b as char).collect() // Latin-1
    };

    let mut c = Cursor { s: &text, i: 0 };
    c.expect('{')?;
    let (mut descr, mut fortran, mut shape) = (None, None, None);
    loop {
        c.ws();
        if c.peek() == Some('}') {
            break;
        }
        let key = c.string()?;
        c.expect(':')?;
        c.ws();
        match key.as_str() {
            "descr" => {
                if matches!(c.peek(), Some('[' | '(')) {
                    return Err(FormatError::StructuredDescr);
                }
                descr = Some(c.string()?);
            }
            "fortran_order" => {
                fortran = Some(match c.word().as_str() {
                    "True" => true,
                    "False" => false,
                    w => return Err(bad(format!("fortran_order must be True or False, found {w:?}"))),
                });
            }
            "shape" => shape = Some(c.shape()?),
            k => return Err(bad(format!("unexpected key {k:?}"))),
        }
        c.ws();
        if c.peek() == Some(',') {
            c.i += 1;
        }
    }
    let descr = descr.ok_or_else(|| bad("missing key 'descr'".into()))?;
    let fortran_order = fortran.ok_or_else(|| bad("missing key 'fortran_order'".into()))?;
    let mut shape = shape.ok_or_else(|| bad("missing key 'shape'".into()))?;
    let dtype = Dtype::from_npy_descr(&descr)?;
    if shape.len() > MAX_RANK {
        return Err(FormatError::RankTooLarge(shape.len()));
    }
    if shape.is_empty() {
        shape.push(1);
    } else if !fortran_order {
        shape.reverse();
    }
    Ok(NpyHeader { dtype, shape, fortran_order, data_offset: end as u64 })
}

/// Builds a `.npy` header for Luma dims `shape`, the inverse of [`parse_npy`]'s mapping (the
/// tuple in the file is `shape` for Fortran order and reversed for C order). Version 1, or 2 when
/// the header does not fit 65535 bytes; padded with spaces and a final newline so that the total
/// is a multiple of 64 (a full 64 when already aligned, as NumPy does).
pub fn write_npy_header(dtype: Dtype, shape: &[usize], fortran_order: bool) -> Vec<u8> {
    let mut dims: Vec<usize> = shape.to_vec();
    if !fortran_order {
        dims.reverse();
    }
    let tuple = match dims.as_slice() {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => format!("({})", dims.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    };
    let dict = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {tuple}, }}",
        dtype.npy_descr(),
        if fortran_order { "True" } else { "False" }
    );
    let framed = |prefix: usize| {
        let pad = 64 - (prefix + dict.len() + 1) % 64;
        dict.len() + pad + 1
    };
    let v1 = framed(10);
    let mut out = Vec::with_capacity(12 + v1);
    out.extend_from_slice(MAGIC);
    let hlen = if v1 <= u16::MAX as usize {
        out.extend_from_slice(&[1, 0]);
        out.extend_from_slice(&(v1 as u16).to_le_bytes());
        v1
    } else {
        let v2 = framed(12);
        out.extend_from_slice(&[2, 0]);
        out.extend_from_slice(&(v2 as u32).to_le_bytes());
        v2
    };
    out.extend_from_slice(dict.as_bytes());
    out.resize(out.len() + hlen - dict.len() - 1, b' ');
    out.push(b'\n');
    out
}

/// Whether a file array is complete or was being written when its sidecar was last updated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarState {
    /// A writer is (or died while) filling the file.
    Writing,
    /// The file holds all its data.
    Complete,
}

/// The `<file>.luma` description of a raw file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sidecar {
    /// Sidecar format version (1).
    pub version: u32,
    /// Element type.
    pub dtype: Dtype,
    /// Luma dims, first dim fastest; unwritten trailing dims are 1.
    pub shape: [usize; 4],
    /// Byte offset of the data in the raw file.
    pub offset: u64,
    /// Whether the data is complete.
    pub state: SidecarState,
}

fn side(m: impl Into<String>) -> FormatError {
    FormatError::BadSidecar(m.into())
}

/// Parses a sidecar: `key = value` lines, `#` comments, unknown keys ignored. `version`, `dtype`,
/// `shape` (one to four dims, padded with 1) and `state` are required; `offset` defaults to 0.
/// `order` and `endian`, when present, must be `F` and `little`.
pub fn parse_sidecar(text: &str) -> Result<Sidecar, FormatError> {
    let (mut version, mut dtype, mut shape, mut offset, mut state) = (None, None, None, 0u64, None);
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (k, v) = line.split_once('=').ok_or_else(|| side(format!("line {}: expected `key = value`", n + 1)))?;
        let (k, v) = (k.trim(), v.trim());
        match k {
            "version" => {
                let x: u32 = v.parse().map_err(|_| side(format!("bad version {v:?}")))?;
                if x != 1 {
                    return Err(side(format!("unsupported version {x}")));
                }
                version = Some(x);
            }
            "dtype" => dtype = Some(Dtype::from_name(v).ok_or_else(|| side(format!("unknown dtype {v:?}")))?),
            "shape" => {
                let dims = v
                    .split(',')
                    .map(|d| d.trim().parse::<usize>().map_err(|_| side(format!("bad shape entry {:?}", d.trim()))))
                    .collect::<Result<Vec<_>, _>>()?;
                if dims.len() > MAX_RANK {
                    return Err(FormatError::RankTooLarge(dims.len()));
                }
                let mut s = [1usize; 4];
                s[..dims.len()].copy_from_slice(&dims);
                shape = Some(s);
            }
            "offset" => offset = v.parse().map_err(|_| side(format!("bad offset {v:?}")))?,
            "order" if v != "F" => return Err(side(format!("order {v:?} is not supported (only F)"))),
            "endian" if v != "little" => {
                return Err(side(format!("endian {v:?} is not supported (only little)")));
            }
            "state" => {
                state = Some(match v {
                    "writing" => SidecarState::Writing,
                    "complete" => SidecarState::Complete,
                    _ => return Err(side(format!("unknown state {v:?}"))),
                });
            }
            _ => {}
        }
    }
    let missing = |k: &str| side(format!("missing key '{k}'"));
    Ok(Sidecar {
        version: version.ok_or_else(|| missing("version"))?,
        dtype: dtype.ok_or_else(|| missing("dtype"))?,
        shape: shape.ok_or_else(|| missing("shape"))?,
        offset,
        state: state.ok_or_else(|| missing("state"))?,
    })
}

/// Renders a sidecar, including `order = F` and `endian = little`. Trailing dims of 1 are omitted
/// (at least one dim is written).
pub fn write_sidecar(s: &Sidecar) -> String {
    let rank = s.shape.iter().rposition(|&d| d != 1).map_or(1, |i| i + 1);
    let dims = s.shape[..rank].iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
    let state = match s.state {
        SidecarState::Writing => "writing",
        SidecarState::Complete => "complete",
    };
    format!(
        "# luma file sidecar\nversion = {}\ndtype = {}\nshape = {dims}\noffset = {}\norder = F\nendian = little\nstate = {state}\n",
        s.version,
        s.dtype.name(),
        s.offset
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-builds a header of the given version around `dict`, padded like NumPy.
    fn build(version: u8, dict: &[u8]) -> Vec<u8> {
        let mut text = dict.to_vec();
        let prefix = if version == 1 { 10 } else { 12 };
        while !(prefix + text.len() + 1).is_multiple_of(64) {
            text.push(b' ');
        }
        text.push(b'\n');
        let mut out = b"\x93NUMPY".to_vec();
        out.extend_from_slice(&[version, 0]);
        if version == 1 {
            out.extend_from_slice(&(text.len() as u16).to_le_bytes());
        } else {
            out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        }
        out.extend_from_slice(&text);
        out
    }

    #[test]
    fn npy_headers_v1_v2_v3_parse() {
        let d = b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
        for v in [1, 2, 3] {
            let mut bytes = build(v, d);
            let len = bytes.len() as u64;
            bytes.extend_from_slice(&[0; 48]); // data follows
            let h = parse_npy(&bytes).unwrap();
            assert_eq!(h.dtype, Dtype::F32);
            assert_eq!(h.shape, [4, 3]); // C order: reversed
            assert!(!h.fortran_order);
            assert_eq!(h.data_offset, len);
        }
        // Fortran order keeps the dims; keys in any order; double quotes; no trailing comma.
        let h = parse_npy(&build(1, br#"{"shape": (3, 4, 5), "fortran_order": True, "descr": "<i2"}"#)).unwrap();
        assert_eq!((h.dtype, h.shape.as_slice(), h.fortran_order), (Dtype::I16, &[3, 4, 5][..], true));
        // () is [1], either order; (n,) with trailing comma.
        for f in ["True", "False"] {
            let d = format!("{{'descr': '|u1', 'fortran_order': {f}, 'shape': (), }}");
            assert_eq!(parse_npy(&build(1, d.as_bytes())).unwrap().shape, [1]);
        }
        let h = parse_npy(&build(1, b"{'descr': '<c8', 'fortran_order': False, 'shape': (7,), }")).unwrap();
        assert_eq!((h.dtype, h.shape.as_slice()), (Dtype::C64, &[7][..]));
        // v3 is UTF-8, v1 Latin-1: a non-ASCII byte in an ignored position (none allowed in keys)
        // must still be read as text, not crash.
        assert!(parse_npy(&build(1, b"{'descr': '<f4', 'fortran_order': False, 'shape': (1,), '\xe9': 1}")).is_err());
        // Truncation and magic.
        let full = build(1, d);
        assert_eq!(parse_npy(&full[..20]), Err(FormatError::Truncated));
        assert_eq!(parse_npy(b"NOTNPY.."), Err(FormatError::BadMagic));
        assert_eq!(parse_npy(b"\x93NUMPY\x09\x00\x00\x00"), Err(FormatError::UnsupportedVersion(9, 0)));
        // Header over 1 MiB.
        let mut big = b"\x93NUMPY\x02\x00".to_vec();
        big.extend_from_slice(&(2u32 << 20).to_le_bytes());
        assert_eq!(parse_npy(&big), Err(FormatError::HeaderTooLarge(2 << 20)));
    }

    #[test]
    fn npy_refuses_structured_object_bool_and_big_endian_dtypes() {
        let with = |descr: &str, shape: &str| {
            let d = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': {shape}, }}");
            parse_npy(&build(1, d.as_bytes()))
        };
        let e = with("[('a', '<f4'), ('b', '<i4')]", "(2,)").unwrap_err();
        assert_eq!(e, FormatError::StructuredDescr);
        assert!(e.to_string().contains("structured"));
        for (d, want) in [("'|O'", "|O"), ("'|b1'", "|b1"), ("'<f16'", "<f16"), ("'<u8'", "<u8"), ("'<M8[ns]'", "M8")] {
            let e = with(d, "(2,)").unwrap_err();
            assert!(matches!(e, FormatError::UnsupportedDtype(_)), "{d}");
            assert!(e.to_string().contains(want), "{e}");
        }
        let e = with("'>f4'", "(2,)").unwrap_err();
        assert_eq!(e, FormatError::BigEndian(">f4".into()));
        assert!(e.to_string().contains("big-endian"));
        let e = with("'<f4'", "(1, 2, 3, 4, 5)").unwrap_err();
        assert_eq!(e, FormatError::RankTooLarge(5));
        assert!(e.to_string().contains("rank 5"));
        assert!(with("'<f4'", "(1, x)").is_err());
        assert!(with("'<f4'", "(-1,)").is_err());
    }

    #[test]
    fn npy_header_is_padded_to_64() {
        for d in Dtype::ALL {
            for shape in [&[1usize][..], &[3, 4], &[100_000, 2, 3, 4], &[usize::MAX / 3, 1]] {
                for f in [false, true] {
                    let h = write_npy_header(d, shape, f);
                    assert_eq!(h.len() % 64, 0);
                    assert_eq!(*h.last().unwrap(), b'\n');
                    let p = parse_npy(&h).unwrap();
                    assert_eq!((p.dtype, p.shape.as_slice(), p.fortran_order), (d, shape, f));
                    assert_eq!(p.data_offset, h.len() as u64);
                }
            }
        }
        // Byte-for-byte what NumPy writes for np.zeros((3, 4), '<f4').
        let h = write_npy_header(Dtype::F32, &[4, 3], false);
        assert_eq!(h.len(), 128);
        assert_eq!(&h[..10], b"\x93NUMPY\x01\x00\x76\x00");
        assert!(h[10..].starts_with(b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }  "));
        // Past 65535 bytes the version becomes 2.
        let shape = vec![1usize; 30_000];
        let h = write_npy_header(Dtype::U8, &shape, true);
        assert_eq!(&h[6..8], &[2, 0]);
        assert_eq!(h.len() % 64, 0);
        assert_eq!(u32::from_le_bytes(h[8..12].try_into().unwrap()) as usize, h.len() - 12);
    }

    #[test]
    fn sidecar_round_trips_and_ignores_unknown_keys() {
        for state in [SidecarState::Writing, SidecarState::Complete] {
            for shape in [[1024, 512, 16, 1], [5, 1, 1, 1], [1, 1, 1, 1], [2, 1, 3, 4], [0, 1, 1, 1]] {
                let s = Sidecar { version: 1, dtype: Dtype::C64, shape, offset: 4096, state };
                let text = write_sidecar(&s);
                assert!(text.contains("order = F") && text.contains("endian = little"));
                assert_eq!(parse_sidecar(&text).unwrap(), s);
            }
        }
        let s = parse_sidecar(
            "# a comment\n\nversion = 1\nfuture = yes\ndtype = f16\nshape = 1024, 512, 16\nstate = writing\nmore stuff = 1\n",
        )
        .unwrap();
        assert_eq!(s.shape, [1024, 512, 16, 1]);
        assert_eq!((s.dtype, s.offset, s.state), (Dtype::F16, 0, SidecarState::Writing));

        let base = "version = 1\ndtype = f32\nshape = 4\nstate = complete\n";
        assert!(parse_sidecar(base).is_ok());
        for (extra, want) in [
            ("order = C\n", "order"),
            ("endian = big\n", "endian"),
            ("dtype = bool\n", "dtype"),
            ("shape = 1, 2, 3, 4, 5\n", "rank"),
            ("state = done\n", "state"),
            ("version = 2\n", "version"),
            ("offset = -1\n", "offset"),
            ("garbage\n", "key = value"),
        ] {
            let e = parse_sidecar(&format!("{base}{extra}")).unwrap_err();
            assert!(e.to_string().contains(want), "{extra}: {e}");
        }
        assert!(parse_sidecar("version = 1\n").unwrap_err().to_string().contains("missing"));
        assert!(parse_sidecar("order = F\nendian = little\nversion = 1\ndtype = u8\nshape = 1\nstate = complete").is_ok());
    }

    #[test]
    fn dtype_names_and_descrs_round_trip() {
        for d in Dtype::ALL {
            assert_eq!(Dtype::from_name(d.name()), Some(d));
            assert_eq!(Dtype::from_npy_descr(d.npy_descr()), Ok(d));
        }
        assert_eq!(Dtype::from_name("f33"), None);
        assert_eq!(Dtype::from_npy_descr("<u1"), Ok(Dtype::U8));
    }
}
