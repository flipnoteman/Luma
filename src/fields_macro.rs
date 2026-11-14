//! The `fields!` macro: declares a record of arrays (see [`Fields`](crate::Fields)).

/// Declares a struct of named fields and implements [`Fields`](crate::Fields) and the record
/// operators for it. Every field is an `Array<T>`, an array or tuple of records, or another
/// record; all of them have the same element type `T` (`f32`, `Df32`, `f64`, `u32`), which
/// becomes `Fields::Elem`. A mismatch is a type error.
///
/// ```
/// use luma::{fields, Array, Context};
///
/// fields! {
///     /// The conserved state of a flow.
///     #[derive(Clone, Debug)]
///     pub struct Cons {
///         pub rho: Array<f32>,
///         pub e: Array<f32>,
///     }
/// }
///
/// fields! {
///     // The element type cannot be read from fields that are not `Array<T>`: name it.
///     #[elem = f32]
///     struct Pair {
///         a: Cons,
///         b: Cons,
///     }
/// }
///
/// # fn main() -> luma::Result<()> {
/// let ctx = Context::new()?;
/// let q = Cons {
///     rho: Array::from_slice_on(&ctx, &[1.0, 2.0], [2])?,
///     e: Array::from_slice_on(&ctx, &[3.0, 4.0], [2])?,
/// };
/// let r = &q * 2.0 + &q;
/// assert_eq!(r.rho.to_vec()?, vec![3.0, 6.0]);
/// # let _ = |p: Pair| p;
/// # Ok(()) }
/// ```
///
/// # Syntax
///
/// ```text
/// luma::fields! {
///     #[elem = T]                 // optional, must come first; see below
///     #[derive(..)] /// docs      // any other attributes, passed through
///     pub struct Name {
///         pub field: Type,        // field visibility is passed through; trailing comma optional
///         ...
///     }
/// }
/// ```
///
/// The element type `T` is the argument of the first field's type when that is written
/// `Array<T>` (the name `Array`, imported from luma). For any other first field (a nested
/// record, `[Array<T>; N]`, a tuple, `luma::Array<T>`) write `#[elem = T]` before the other
/// attributes. (A declarative macro cannot look inside a type, and the operators have to name
/// the element type: a projection in an impl header makes them overlap as far as coherence can
/// tell.)
///
/// Generic structs are not supported (the macro takes `struct Name { .. }` without parameters
/// or a `where` clause), nor tuple structs, nor attributes on fields. A record has at least one
/// field.
///
/// # Generated
///
/// - The struct itself, with the given attributes and visibilities.
/// - `impl Fields`: `COUNT` is the sum of the fields' counts; `each`, `each_mut` and `build`
///   visit the fields in declaration order.
/// - The operators `+ - * /` between two records, a record and an element value (`f32` for
///   `f32` records), and a record and an `Array<Elem>` (the array applies to every field), in
///   every reference/owned combination and with the scalar or array on either side; `+= -= *= /=`
///   with `&Record`, an element value and `&Array<Elem>`; and unary `-` on `&Record` and `Record`
///   (not for `u32` records). Every operator borrows its operands and forwards to the hidden
///   methods of [`Fields`](crate::Fields).
#[macro_export]
macro_rules! fields {
    // Explicit element type; `u32` has no negation, and a `ty` fragment cannot be matched
    // against it later, so it gets its own rules.
    (#[elem = u32] $($rest:tt)*) => {
        $crate::fields!(@parse [u32] [no] $($rest)*);
    };
    (#[elem = $elem:ty] $($rest:tt)*) => {
        $crate::fields!(@parse [$elem] [neg] $($rest)*);
    };
    // Inferred from the first field.
    ($(#[$m:meta])* $vis:vis struct $name:ident { $fv:vis $f:ident : Array<u32> $($rest:tt)* }) => {
        $crate::fields!(@parse [u32] [no] $(#[$m])* $vis struct $name { $fv $f : Array<u32> $($rest)* });
    };
    ($(#[$m:meta])* $vis:vis struct $name:ident { $fv:vis $f:ident : Array<$elem:ty> $($rest:tt)* }) => {
        $crate::fields!(@parse [$elem] [neg] $(#[$m])* $vis struct $name { $fv $f : Array<$elem> $($rest)* });
    };

    // The struct, with the element type and negation flag already decided.
    (@parse [$elem:ty] [$neg:ident] $(#[$m:meta])* $vis:vis struct $name:ident {
        $( $fv:vis $f:ident : $ty:ty ),+ $(,)?
    }) => {
        $(#[$m])*
        $vis struct $name {
            $( $fv $f: $ty, )+
        }
        $crate::fields!(@impl $name $elem; $( $f: $ty ),+);
        $crate::fields!(@neg [$neg] $name);
        $crate::fields!(@op $name $elem; Add add AddAssign add_assign);
        $crate::fields!(@op $name $elem; Sub sub SubAssign sub_assign);
        $crate::fields!(@op $name $elem; Mul mul MulAssign mul_assign);
        $crate::fields!(@op $name $elem; Div div DivAssign div_assign);
    };

    (@impl $name:ident $elem:ty; $( $f:ident : $ty:ty ),+) => {
        impl $crate::Fields for $name {
            type Elem = $elem;
            const COUNT: usize = 0 $( + <$ty as $crate::Fields>::COUNT )+;

            fn each<'a>(&'a self, f: &mut dyn ::core::ops::FnMut(&'a $crate::Array<Self::Elem>)) {
                $( <$ty as $crate::Fields>::each(&self.$f, f); )+
            }

            fn each_mut<'a>(&'a mut self, f: &mut dyn ::core::ops::FnMut(&'a mut $crate::Array<Self::Elem>)) {
                $( <$ty as $crate::Fields>::each_mut(&mut self.$f, f); )+
            }

            fn build(next: &mut dyn ::core::ops::FnMut() -> $crate::Array<Self::Elem>) -> Self {
                Self { $( $f: <$ty as $crate::Fields>::build(next), )+ }
            }
        }
    };

    // Unsigned arrays have no negation.
    (@neg [no] $name:ident) => {};
    (@neg [neg] $name:ident) => {
        impl ::core::ops::Neg for &$name {
            type Output = $name;
            #[track_caller]
            fn neg(self) -> $name { $crate::Fields::negate(self) }
        }
        impl ::core::ops::Neg for $name {
            type Output = $name;
            #[track_caller]
            fn neg(self) -> $name { $crate::Fields::negate(&self) }
        }
    };

    (@op $name:ident $elem:ty; $variant:ident $method:ident $assign_trait:ident $assign_method:ident) => {
        // record op record
        impl ::core::ops::$variant<$name> for $name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $name) -> $name {
                $crate::Fields::binary(&self, &rhs, $crate::__private::FieldOp::$variant)
            }
        }
        impl ::core::ops::$variant<&$name> for $name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$name) -> $name {
                $crate::Fields::binary(&self, rhs, $crate::__private::FieldOp::$variant)
            }
        }
        impl ::core::ops::$variant<$name> for &$name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $name) -> $name {
                $crate::Fields::binary(self, &rhs, $crate::__private::FieldOp::$variant)
            }
        }
        impl ::core::ops::$variant<&$name> for &$name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$name) -> $name {
                $crate::Fields::binary(self, rhs, $crate::__private::FieldOp::$variant)
            }
        }

        // record op scalar
        impl ::core::ops::$variant<$elem> for $name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $elem) -> $name {
                $crate::Fields::binary_scalar(&self, rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        impl ::core::ops::$variant<$elem> for &$name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $elem) -> $name {
                $crate::Fields::binary_scalar(self, rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        // scalar op record
        impl ::core::ops::$variant<$name> for $elem {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $name) -> $name {
                $crate::Fields::binary_scalar(&rhs, self, $crate::__private::FieldOp::$variant, true)
            }
        }
        impl ::core::ops::$variant<&$name> for $elem {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$name) -> $name {
                $crate::Fields::binary_scalar(rhs, self, $crate::__private::FieldOp::$variant, true)
            }
        }

        // record op array
        impl ::core::ops::$variant<$crate::Array<$elem>> for $name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $crate::Array<$elem>) -> $name {
                $crate::Fields::binary_array(&self, &rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        impl ::core::ops::$variant<&$crate::Array<$elem>> for $name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$crate::Array<$elem>) -> $name {
                $crate::Fields::binary_array(&self, rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        impl ::core::ops::$variant<$crate::Array<$elem>> for &$name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $crate::Array<$elem>) -> $name {
                $crate::Fields::binary_array(self, &rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        impl ::core::ops::$variant<&$crate::Array<$elem>> for &$name {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$crate::Array<$elem>) -> $name {
                $crate::Fields::binary_array(self, rhs, $crate::__private::FieldOp::$variant, false)
            }
        }
        // array op record
        impl ::core::ops::$variant<$name> for $crate::Array<$elem> {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $name) -> $name {
                $crate::Fields::binary_array(&rhs, &self, $crate::__private::FieldOp::$variant, true)
            }
        }
        impl ::core::ops::$variant<&$name> for $crate::Array<$elem> {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$name) -> $name {
                $crate::Fields::binary_array(rhs, &self, $crate::__private::FieldOp::$variant, true)
            }
        }
        impl ::core::ops::$variant<$name> for &$crate::Array<$elem> {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: $name) -> $name {
                $crate::Fields::binary_array(&rhs, self, $crate::__private::FieldOp::$variant, true)
            }
        }
        impl ::core::ops::$variant<&$name> for &$crate::Array<$elem> {
            type Output = $name;
            #[track_caller]
            fn $method(self, rhs: &$name) -> $name {
                $crate::Fields::binary_array(rhs, self, $crate::__private::FieldOp::$variant, true)
            }
        }

        // compound assignment
        impl ::core::ops::$assign_trait<&$name> for $name {
            #[track_caller]
            fn $assign_method(&mut self, rhs: &$name) {
                $crate::Fields::assign_op(self, rhs, $crate::__private::FieldOp::$variant);
            }
        }
        impl ::core::ops::$assign_trait<$elem> for $name {
            #[track_caller]
            fn $assign_method(&mut self, rhs: $elem) {
                $crate::Fields::assign_op_scalar(self, rhs, $crate::__private::FieldOp::$variant);
            }
        }
        impl ::core::ops::$assign_trait<&$crate::Array<$elem>> for $name {
            #[track_caller]
            fn $assign_method(&mut self, rhs: &$crate::Array<$elem>) {
                $crate::Fields::assign_op_array(self, rhs, $crate::__private::FieldOp::$variant);
            }
        }
    };

    ($($tt:tt)*) => {
        ::core::compile_error!(
            "luma::fields! expects `[#[elem = T]] [attributes] [pub] struct Name { [pub] field: Type, .. }`; \
             without `#[elem = T]` the first field must be `Array<T>`"
        );
    };
}
