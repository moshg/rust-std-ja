// edition:2018
#![feature(async_await)]

use std::io::Error;

fn make_unit() -> Result<(), Error> {
    Ok(())
}

fn main() {
    let fut = async {
        make_unit()?; //~ ERROR type annotations needed

        Ok(())
    };
}
