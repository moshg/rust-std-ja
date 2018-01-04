// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


pub struct CrateId {
    local_path: String,
    junk: String
}

impl CrateId {
    fn new(s: &str) -> CrateId {
        CrateId {
            local_path: s.to_string(),
            junk: "wutevs".to_string()
        }
    }
}

pub fn remove_package_from_database() {
    let mut lines_to_use: Vec<&CrateId> = Vec::new();
        //~^ NOTE cannot infer an appropriate lifetime
    let push_id = |installed_id: &CrateId| {
        //~^ NOTE borrowed data cannot outlive this closure
        lines_to_use.push(installed_id);
        //~^ ERROR borrowed data cannot be moved outside of its closure
        //~| NOTE cannot be moved outside of its closure
    };
    list_database(push_id);

    for l in &lines_to_use {
        println!("{}", l.local_path);
    }

}

pub fn list_database<F>(mut f: F) where F: FnMut(&CrateId) {
    let stuff = ["foo", "bar"];

    for l in &stuff {
        f(&CrateId::new(*l));
    }
}

pub fn main() {
    remove_package_from_database();
}
