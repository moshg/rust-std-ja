// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module defines a generic file format that allows to check if a given
//! file generated by incremental compilation was generated by a compatible
//! compiler version. This file format is used for the on-disk version of the
//! dependency graph and the exported metadata hashes.
//!
//! In practice "compatible compiler version" means "exactly the same compiler
//! version", since the header encodes the git commit hash of the compiler.
//! Since we can always just ignore the incremental compilation cache and
//! compiler versions don't change frequently for the typical user, being
//! conservative here practically has no downside.

use std::io::{self, Read};
use std::path::Path;
use std::fs::File;
use std::env;

use rustc::session::config::nightly_options;

/// The first few bytes of files generated by incremental compilation
const FILE_MAGIC: &'static [u8] = b"RSIC";

/// Change this if the header format changes
const HEADER_FORMAT_VERSION: u16 = 0;

/// A version string that hopefully is always different for compiler versions
/// with different encodings of incremental compilation artifacts. Contains
/// the git commit hash.
const RUSTC_VERSION: Option<&'static str> = option_env!("CFG_VERSION");

pub fn write_file_header<W: io::Write>(stream: &mut W) -> io::Result<()> {
    stream.write_all(FILE_MAGIC)?;
    stream.write_all(&[(HEADER_FORMAT_VERSION >> 0) as u8,
                       (HEADER_FORMAT_VERSION >> 8) as u8])?;

    let rustc_version = rustc_version();
    assert_eq!(rustc_version.len(), (rustc_version.len() as u8) as usize);
    stream.write_all(&[rustc_version.len() as u8])?;
    stream.write_all(rustc_version.as_bytes())?;

    Ok(())
}

/// Reads the contents of a file with a file header as defined in this module.
///
/// - Returns `Ok(Some(data, pos))` if the file existed and was generated by a
///   compatible compiler version. `data` is the entire contents of the file
///   and `pos` points to the first byte after the header.
/// - Returns `Ok(None)` if the file did not exist or was generated by an
///   incompatible version of the compiler.
/// - Returns `Err(..)` if some kind of IO error occurred while reading the
///   file.
pub fn read_file(report_incremental_info: bool, path: &Path)
    -> io::Result<Option<(Vec<u8>, usize)>>
{
    if !path.exists() {
        return Ok(None);
    }

    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len() as usize;

    let mut data = Vec::with_capacity(file_size);
    file.read_to_end(&mut data)?;

    let mut file = io::Cursor::new(data);

    // Check FILE_MAGIC
    {
        debug_assert!(FILE_MAGIC.len() == 4);
        let mut file_magic = [0u8; 4];
        file.read_exact(&mut file_magic)?;
        if file_magic != FILE_MAGIC {
            report_format_mismatch(report_incremental_info, path, "Wrong FILE_MAGIC");
            return Ok(None)
        }
    }

    // Check HEADER_FORMAT_VERSION
    {
        debug_assert!(::std::mem::size_of_val(&HEADER_FORMAT_VERSION) == 2);
        let mut header_format_version = [0u8; 2];
        file.read_exact(&mut header_format_version)?;
        let header_format_version = (header_format_version[0] as u16) |
                                    ((header_format_version[1] as u16) << 8);

        if header_format_version != HEADER_FORMAT_VERSION {
            report_format_mismatch(report_incremental_info, path, "Wrong HEADER_FORMAT_VERSION");
            return Ok(None)
        }
    }

    // Check RUSTC_VERSION
    {
        let mut rustc_version_str_len = [0u8; 1];
        file.read_exact(&mut rustc_version_str_len)?;
        let rustc_version_str_len = rustc_version_str_len[0] as usize;
        let mut buffer = Vec::with_capacity(rustc_version_str_len);
        buffer.resize(rustc_version_str_len, 0);
        file.read_exact(&mut buffer)?;

        if buffer != rustc_version().as_bytes() {
            report_format_mismatch(report_incremental_info, path, "Different compiler version");
            return Ok(None);
        }
    }

    let post_header_start_pos = file.position() as usize;
    Ok(Some((file.into_inner(), post_header_start_pos)))
}

fn report_format_mismatch(report_incremental_info: bool, file: &Path, message: &str) {
    debug!("read_file: {}", message);

    if report_incremental_info {
        println!("[incremental] ignoring cache artifact `{}`: {}",
                  file.file_name().unwrap().to_string_lossy(),
                  message);
    }
}

fn rustc_version() -> String {
    if nightly_options::is_nightly_build() {
        if let Some(val) = env::var_os("RUSTC_FORCE_INCR_COMP_ARTIFACT_HEADER") {
            return val.to_string_lossy().into_owned()
        }
    }

    RUSTC_VERSION.expect("Cannot use rustc without explicit version for \
                          incremental compilation")
                 .to_string()
}
