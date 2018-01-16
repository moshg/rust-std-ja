// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Like the `long-linker-command-lines` test this test attempts to blow
// a command line limit for running the linker. Unlike that test, however,
// this test is testing `cmd.exe` specifically rather than the OS.
//
// Unfortunately `cmd.exe` has a 8192 limit which is relatively small
// in the grand scheme of things and anyone sripting rustc's linker
// is probably using a `*.bat` script and is likely to hit this limit.
//
// This test uses a `foo.bat` script as the linker which just simply
// delegates back to this program. The compiler should use a lower
// limit for arguments before passing everything via `@`, which
// means that everything should still succeed here.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write, Read};
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if !cfg!(windows) {
        return
    }

    let tmpdir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let ok = tmpdir.join("ok");
    let not_ok = tmpdir.join("not_ok");
    if env::var("YOU_ARE_A_LINKER").is_ok() {
        match env::args().find(|a| a.contains("@")) {
            Some(file) => { fs::copy(&file[1..], &ok).unwrap(); }
            None => { File::create(&not_ok).unwrap(); }
        }
        return
    }

    let rustc = env::var_os("RUSTC").unwrap_or("rustc".into());
    let me = env::current_exe().unwrap();
    let bat = me.parent()
        .unwrap()
        .join("foo.bat");
    let bat_linker = format!("linker={}", bat.display());
    for i in (1..).map(|i| i * 10) {
        println!("attempt: {}", i);

        let file = tmpdir.join("bar.rs");
        let mut f = BufWriter::new(File::create(&file).unwrap());
        let mut lib_name = String::new();
        for _ in 0..i {
            lib_name.push_str("foo");
        }
        for j in 0..i {
            writeln!(f, "#[link(name = \"{}{}\")]", lib_name, j).unwrap();
        }
        writeln!(f, "extern {{}}\nfn main() {{}}").unwrap();
        f.into_inner().unwrap();

        drop(fs::remove_file(&ok));
        drop(fs::remove_file(&not_ok));
        let status = Command::new(&rustc)
            .arg(&file)
            .arg("-C").arg(&bat_linker)
            .arg("--out-dir").arg(&tmpdir)
            .env("YOU_ARE_A_LINKER", "1")
            .env("MY_LINKER", &me)
            .status()
            .unwrap();

        if !status.success() {
            panic!("rustc didn't succeed: {}", status);
        }

        if !ok.exists() {
            assert!(not_ok.exists());
            continue
        }

        let mut contents = String::new();
        File::open(&ok).unwrap().read_to_string(&mut contents).unwrap();

        for j in 0..i {
            assert!(contents.contains(&format!("{}{}", lib_name, j)));
        }

        break
    }
}
