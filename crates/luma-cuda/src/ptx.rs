//! The text of a PTX kernel under construction, and the splice of an epilogue hook into it.

pub(crate) struct Ptx {
    pub(crate) text: String,
}

impl Ptx {
    pub(crate) fn line(&mut self, s: impl AsRef<str>) {
        self.text.push_str("    ");
        self.text.push_str(s.as_ref());
        self.text.push('\n');
    }
}

/// The `.func luma_epi` of the PTX NVVM printed for an epilogue hook (see `codegen::emit_hook`), ready to be expanded at each store
/// of a product's kernel, with the `.version` and `.target` lines of that PTX (the product's kernel takes them: the instructions
/// the hook uses may need a newer ISA than the product's text names).
#[derive(Clone, Debug)]
pub(crate) struct Hook {
    /// The body of the function: the lines between its braces.
    body: Vec<String>,
    /// `.version` line.
    pub(crate) version: String,
    /// `.target` line.
    pub(crate) target: String,
    /// Extra operands the hook reads (parameters `p0..` after `n` and the scalars).
    pub(crate) operands: u32,
    /// Scalars: 32-bit words of the uniform after the eight of the product.
    pub(crate) scalars: u32,
}

impl Hook {
    /// The hook of the PTX text `ptx` of NVVM, `Err` if it is not one the expander takes: more than the function (a global table,
    /// local memory, a call) or a form of its text the expander does not know.
    pub(crate) fn parse(ptx: &str, operands: u32, scalars: u32) -> Result<Hook, String> {
        let line = |prefix: &str| ptx.lines().find(|l| l.starts_with(prefix)).map(|l| l.trim().to_owned());
        let (Some(version), Some(target)) = (line(".version"), line(".target")) else { return Err("the hook's PTX has no .version or .target".into()) };
        let at = ptx.find("luma_epi(").ok_or("the hook's PTX has no luma_epi")?;
        let open = at + ptx[at..].find("\n{\n").ok_or("the hook's PTX has no function body")? + 3;
        let close = open + ptx[open..].find("\n}").ok_or("the hook's PTX has an unfinished function")?;
        // Anything the module holds besides the function (data tables) would have to move into the product's module.
        if ptx[..at].lines().any(|l| l.starts_with(".global") || l.starts_with(".const") || l.starts_with(".extern")) || ptx[close..].contains(".global") {
            return Err("the hook's PTX has global data".into());
        }
        let body: Vec<String> = ptx[open..close].lines().map(|l| l.trim().to_owned()).filter(|l| !l.is_empty()).collect();
        if body.iter().any(|l| l.starts_with("call") || l.contains(".local") || l.contains(".shared") || l.starts_with("ld.param.u8")) {
            return Err("the hook's PTX calls a function or uses local memory".into());
        }
        Ok(Hook { body, version, target, operands, scalars })
    }

    /// The product's PTX `text` with this hook's `.version` and `.target`.
    pub(crate) fn reconcile(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut done = (false, false);
        for l in text.lines() {
            if !done.0 && l.starts_with(".version") {
                out.push_str(&self.version);
                done.0 = true;
            } else if !done.1 && l.starts_with(".target") {
                out.push_str(&self.target);
                done.1 = true;
            } else {
                out.push_str(l);
            }
            out.push('\n');
        }
        out
    }

    /// The hook applied to `res` (a register of the accumulator) in a block of its own at site `site`; `args` are the registers of the
    /// parameters in order with their widths, `acc` is `args[0]` and `res` receives the result. The body is the function's: its
    /// registers renamed with the prefix `%hi{site}`, `ld.param` of parameter `k` a `mov` from `args[k]` (a `cvt` of its low bits when the
    /// function reads fewer bits: NVVM narrows a `trunc i64 to i32` into a 32-bit parameter load), `st.param` of the result a `mov`
    /// to `res`, `ret` a branch to the end (nothing when it is the last statement).
    pub(crate) fn expand(&self, site: u32, args: &[(String, u32)], res: &str) -> Result<String, String> {
        let lines = &self.body;
        let mut names: Vec<String> = Vec::new();
        for l in lines {
            if let Some(rest) = l.strip_prefix(".reg") {
                let name = rest.split_whitespace().nth(1).unwrap_or("").trim_end_matches(';');
                names.push(name.split('<').next().unwrap_or("").trim_start_matches('%').to_owned());
            }
        }
        let rename = |s: &str| -> String {
            let mut out = String::with_capacity(s.len() + 8);
            let b = s.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'%' || b[i] == b'$' {
                    let start = i;
                    i += 1;
                    while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                        i += 1;
                    }
                    let id = &s[start + 1..i];
                    if b[start] == b'$' {
                        out.push_str(&format!("$i{site}{id}"));
                    } else if names.iter().any(|n| id.trim_end_matches(|c: char| c.is_ascii_digit()) == n) {
                        out.push_str(&format!("%hi{site}{id}"));
                    } else {
                        out.push('%');
                        out.push_str(id);
                    }
                } else {
                    out.push(b[i] as char);
                    i += 1;
                }
            }
            out
        };
        let mut s = String::from("    {\n");
        let last = lines.len().saturating_sub(1);
        let mut needs_end = false;
        for (n, l) in lines.iter().enumerate() {
            if let Some(rest) = l.strip_prefix(".reg") {
                let mut it = rest.split_whitespace();
                let (Some(ty), Some(decl)) = (it.next(), it.next()) else { return Err(format!("a register declaration the expander does not know: {l}")) };
                s.push_str(&format!("    .reg {ty} %hi{site}{};\n", decl.trim_end_matches(';').trim_start_matches('%')));
            } else if let Some(rest) = l.strip_prefix("ld.param.") {
                let (ty, rest) = rest.split_once(' ').ok_or("a parameter load the expander does not know")?;
                let (dst, src) = rest.split_once(',').ok_or("a parameter load the expander does not know")?;
                let k: usize = src.trim().trim_start_matches('[').trim_end_matches("];").rsplit('_').next().and_then(|k| k.parse().ok()).ok_or("a parameter the expander does not know")?;
                let bits: u32 = ty.trim_start_matches(|c: char| c.is_ascii_alphabetic()).parse().map_err(|_| "a parameter type the expander does not know")?;
                let (arg, arg_bits) = args.get(k).ok_or("the hook reads a parameter it does not have")?;
                if bits == *arg_bits {
                    s.push_str(&format!("    mov.{ty} {}, {arg};\n", rename(dst.trim())));
                } else if bits < *arg_bits {
                    s.push_str(&format!("    cvt.u{bits}.u{arg_bits} {}, {arg};\n", rename(dst.trim())));
                } else {
                    return Err(format!("a {bits}-bit read of a {arg_bits}-bit argument"));
                }
            } else if let Some(rest) = l.strip_prefix("st.param.") {
                let (ty, rest) = rest.split_once(' ').ok_or("a result store the expander does not know")?;
                let (_, src) = rest.split_once(',').ok_or("a result store the expander does not know")?;
                s.push_str(&format!("    mov.{ty} {res}, {};\n", rename(src.trim().trim_end_matches(';'))));
            } else if l == "ret;" {
                if n != last {
                    s.push_str(&format!("    bra $i{site}end;\n"));
                    needs_end = true;
                }
            } else if l.starts_with('{') || l.starts_with('}') || l.starts_with(".param") {
                return Err(format!("a nested scope or a parameter in the hook: {l}"));
            } else {
                s.push_str(&format!("    {}\n", rename(l)));
            }
        }
        if needs_end {
            s.push_str(&format!("$i{site}end:\n"));
        }
        s.push_str("    }\n");
        Ok(s)
    }
}

/// Appends one formatted instruction to a [`Ptx`].
macro_rules! p {
    ($w:expr, $($arg:tt)*) => { $w.line(format!($($arg)*)) };
}
