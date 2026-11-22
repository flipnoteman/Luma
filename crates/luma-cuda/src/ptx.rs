//! The text of a PTX kernel under construction.

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

/// Appends one formatted instruction to a [`Ptx`].
macro_rules! p {
    ($w:expr, $($arg:tt)*) => { $w.line(format!($($arg)*)) };
}
