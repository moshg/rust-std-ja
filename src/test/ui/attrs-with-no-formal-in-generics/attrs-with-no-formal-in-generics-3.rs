// This test checks variations on `<#[attr] 'a, #[oops]>`, where
// `#[oops]` is left dangling (that is, it is unattached, with no
// formal binding following it).

struct RefIntPair<'a, 'b>(&'a u32, &'b u32);

fn hof_lt<Q>(_: Q)
    where Q: for <#[rustc_1] 'a, 'b, #[oops]> Fn(RefIntPair<'a,'b>) -> &'b u32
    //~^ ERROR trailing attribute after lifetime parameters
    //~| ERROR unless otherwise specified, attributes with the prefix `rustc_` are reserved for
{
}

fn main() {
}
