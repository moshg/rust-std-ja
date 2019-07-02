// edition:2018
// run-pass

// Test that we can use async fns with multiple arbitrary lifetimes.

#![feature(async_await)]

async fn multiple_named_lifetimes<'a, 'b>(_: &'a u8, _: &'b u8) {}

fn main() {
    let _ = multiple_named_lifetimes(&22, &44);
}
