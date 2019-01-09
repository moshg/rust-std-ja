// compile-flags: -Z force-unstable-if-unmarked

// @matches internal/index.html '//*[@class="docblock-short"]' \
//      '^\[Internal\] Docs'
// @has internal/struct.S.html '//*[@class="stab internal"]' \
//      'This is an internal compiler API. (rustc_private)'
/// Docs
pub struct S;

fn main() {}
