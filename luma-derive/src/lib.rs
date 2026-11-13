//! Derive macro for `luma::Fields`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::{Data, DeriveInput, Fields, GenericArgument, PathArguments, Type, parse_macro_input};

/// Derives `luma::Fields` for a struct of named fields, each of which implements `Fields`
/// (`Array<T>`, arrays or tuples of records, nested records). All the fields must have the same element
/// type `T` (`f32`, `Df32`, `f64`, `u32`), which becomes `Fields::Elem` of the record; a mismatch is a type error.
///
/// The element type is read from the field types: the first `Array<T>` named in them (also inside
/// arrays, tuples and references). A record whose fields are all other records has none to read:
/// give it with `#[fields(elem = T)]` (the default is `f32`).
///
/// Generates:
/// - `impl luma::Fields`: `COUNT` is the sum of the fields' counts; `each` and `build` visit the fields in
///   declaration order.
/// - The operators `+ - * /` between two records, a record and an element value (`f32` for `f32` records), and
///   a record and an `Array<Elem>` (the array applies to every field), in every reference/owned combination
///   and with the scalar or array on either side; `+= -= *= /=` with `&Record`, an element value and
///   `&Array<Elem>`; and unary `-` on `&Record` and `Record` (not for `u32` records).
///   Every operator borrows its operands and forwards to the hidden methods of `luma::Fields`.
#[proc_macro_derive(Fields, attributes(fields))]
pub fn derive_fields(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(&input).into()
}

/// The element type of the record: `#[fields(elem = T)]`, else the type argument of the first
/// `Array<T>` among the field types, else `f32`. (The operators have to name the type: an
/// associated-type projection in an impl header makes them overlap as far as coherence can tell.)
fn element_type(input: &DeriveInput, types: &[&Type]) -> syn::Result<Type> {
    let mut explicit = None;
    for attr in input.attrs.iter().filter(|attr| attr.path().is_ident("fields")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("elem") {
                explicit = Some(meta.value()?.parse::<Type>()?);
                Ok(())
            } else {
                Err(meta.error("expected `elem = Type`"))
            }
        })?;
    }
    if let Some(elem) = explicit {
        return Ok(elem);
    }
    Ok(types.iter().find_map(|ty| array_element(ty)).unwrap_or_else(|| syn::parse_quote!(f32)))
}

/// `T` of the first `Array<T>` named in `ty`.
fn array_element(ty: &Type) -> Option<Type> {
    match ty {
        Type::Path(path) => {
            let last = path.path.segments.last()?;
            let PathArguments::AngleBracketed(args) = &last.arguments else { return None };
            let mut inner = args.args.iter().filter_map(|arg| match arg {
                GenericArgument::Type(ty) => Some(ty),
                _ => None,
            });
            if last.ident == "Array" {
                inner.next().cloned()
            } else {
                inner.find_map(array_element)
            }
        }
        Type::Array(array) => array_element(&array.elem),
        Type::Tuple(tuple) => tuple.elems.iter().find_map(array_element),
        Type::Reference(reference) => array_element(&reference.elem),
        Type::Paren(paren) => array_element(&paren.elem),
        Type::Group(group) => array_element(&group.elem),
        _ => None,
    }
}

fn expand(input: &DeriveInput) -> Tokens {
    let named = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(n) if input.generics.params.is_empty() && input.generics.where_clause.is_none() => {
                Some(n)
            }
            _ => None,
        },
        _ => None,
    };
    let Some(named) = named else {
        return syn::Error::new_spanned(
            &input.ident,
            "#[derive(Fields)] needs a struct with named fields and no generics",
        )
        .to_compile_error();
    };

    let name = &input.ident;
    let names: Vec<_> = named.named.iter().map(|f| f.ident.as_ref().unwrap()).collect();
    let types: Vec<_> = named.named.iter().map(|f| &f.ty).collect();

    if types.is_empty() {
        return syn::Error::new_spanned(&input.ident, "#[derive(Fields)] needs at least one field").to_compile_error();
    }
    let elem = match element_type(input, &types) {
        Ok(elem) => elem,
        Err(error) => return error.to_compile_error(),
    };
    let mut out = quote! {
        impl ::luma::Fields for #name {
            type Elem = #elem;
            const COUNT: usize = 0 #( + <#types as ::luma::Fields>::COUNT )*;

            fn each<'a>(&'a self, f: &mut dyn ::core::ops::FnMut(&'a ::luma::Array<Self::Elem>)) {
                #( <#types as ::luma::Fields>::each(&self.#names, f); )*
            }

            fn each_mut<'a>(&'a mut self, f: &mut dyn ::core::ops::FnMut(&'a mut ::luma::Array<Self::Elem>)) {
                #( <#types as ::luma::Fields>::each_mut(&mut self.#names, f); )*
            }

            fn build(next: &mut dyn ::core::ops::FnMut() -> ::luma::Array<Self::Elem>) -> Self {
                Self { #( #names: <#types as ::luma::Fields>::build(next), )* }
            }
        }
    };

    let ops = [
        ("Add", "add", "add_assign", "AddAssign"),
        ("Sub", "sub", "sub_assign", "SubAssign"),
        ("Mul", "mul", "mul_assign", "MulAssign"),
        ("Div", "div", "div_assign", "DivAssign"),
    ];
    let elem = quote! { #elem };
    let arr = quote! { ::luma::Array<#elem> };

    for (variant, method, assign_method, assign_trait) in ops {
        let variant = format_ident!("{variant}");
        let method = format_ident!("{method}");
        let assign_method = format_ident!("{assign_method}");
        let assign_trait = format_ident!("{assign_trait}");
        let op = quote! { ::luma::__private::FieldOp::#variant };
        let trait_path = quote! { ::core::ops::#variant };
        let assign_path = quote! { ::core::ops::#assign_trait };

        let binary = |lhs: Tokens, rhs: Tokens, body: Tokens| {
            quote! {
                impl #trait_path<#rhs> for #lhs {
                    type Output = #name;
                    #[track_caller]
                    fn #method(self, rhs: #rhs) -> #name { #body }
                }
            }
        };
        // `(type, owned)`: an owned operand is borrowed in the body, a reference passed on.
        let rec = (quote! { #name }, true);
        let rec_ref = (quote! { &#name }, false);
        let arr_t = (quote! { #arr }, true);
        let arr_ref = (quote! { &#arr }, false);
        let lend = |owned: bool, id: Tokens| if owned { quote! { &#id } } else { id };
        let slf = quote! { self };
        let rhs = quote! { rhs };

        // record op record
        for l in [&rec, &rec_ref] {
            for r in [&rec, &rec_ref] {
                let (a, b) = (lend(l.1, slf.clone()), lend(r.1, rhs.clone()));
                out.extend(binary(l.0.clone(), r.0.clone(), quote! { ::luma::Fields::binary(#a, #b, #op) }));
            }
        }

        // record op scalar, scalar op record
        for l in [&rec, &rec_ref] {
            let a = lend(l.1, slf.clone());
            out.extend(binary(l.0.clone(), elem.clone(), quote! { ::luma::Fields::binary_scalar(#a, rhs, #op, false) }));
        }
        for r in [&rec, &rec_ref] {
            let b = lend(r.1, rhs.clone());
            out.extend(binary(elem.clone(), r.0.clone(), quote! { ::luma::Fields::binary_scalar(#b, self, #op, true) }));
        }

        // record op array, array op record
        for l in [&rec, &rec_ref] {
            for r in [&arr_t, &arr_ref] {
                let (a, b) = (lend(l.1, slf.clone()), lend(r.1, rhs.clone()));
                out.extend(binary(l.0.clone(), r.0.clone(), quote! { ::luma::Fields::binary_array(#a, #b, #op, false) }));
            }
        }
        for l in [&arr_t, &arr_ref] {
            for r in [&rec, &rec_ref] {
                let (a, b) = (lend(l.1, slf.clone()), lend(r.1, rhs.clone()));
                out.extend(binary(l.0.clone(), r.0.clone(), quote! { ::luma::Fields::binary_array(#b, #a, #op, true) }));
            }
        }

        // compound assignment
        out.extend(quote! {
            impl #assign_path<&#name> for #name {
                #[track_caller]
                fn #assign_method(&mut self, rhs: &#name) {
                    *self = ::luma::Fields::binary(&*self, rhs, #op);
                }
            }
            impl #assign_path<#elem> for #name {
                #[track_caller]
                fn #assign_method(&mut self, rhs: #elem) {
                    *self = ::luma::Fields::binary_scalar(&*self, rhs, #op, false);
                }
            }
            impl #assign_path<&#arr> for #name {
                #[track_caller]
                fn #assign_method(&mut self, rhs: &#arr) {
                    *self = ::luma::Fields::binary_array(&*self, rhs, #op, false);
                }
            }
        });
    }

    // Unsigned arrays have no negation.
    if elem.to_string() == "u32" {
        return out;
    }
    out.extend(quote! {
        impl ::core::ops::Neg for &#name {
            type Output = #name;
            #[track_caller]
            fn neg(self) -> #name { ::luma::Fields::negate(self) }
        }
        impl ::core::ops::Neg for #name {
            type Output = #name;
            #[track_caller]
            fn neg(self) -> #name { ::luma::Fields::negate(&self) }
        }
    });
    out
}
