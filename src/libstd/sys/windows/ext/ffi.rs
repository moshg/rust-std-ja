// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Windows-specific extensions to the primitives in the `std::ffi` module.
//!
//! # Overview
//!
//! For historical reasons, the Windows API uses a form of potentially
//! ill-formed UTF-16 encoding for strings.  Specifically, the 16-bit
//! code units in Windows strings may contain [isolated surrogate code
//! points which are not paired together][ill-formed-utf-16].  The
//! Unicode standard requires that surrogate code points (those in the
//! range U+D800 to U+DFFF) always be *paired*, because in the UTF-16
//! encoding a *surrogate code unit pair* is used to encode a single
//! character.  For compatibility with code that does not enforce
//! these pairings, Windows does not enforce them, either.
//!
//! While it is not always possible to convert such a string losslessly into
//! a valid UTF-16 string (or even UTF-8), it is often desirable to be
//! able to round-trip such a string from and to Windows APIs
//! losslessly.  For example, some Rust code may be "bridging" some
//! Windows APIs together, just passing `WCHAR` strings among those
//! APIs without ever really looking into the strings.
//!
//! If Rust code *does* need to look into those strings, it can
//! convert them to valid UTF-8, possibly lossily, by substituting
//! invalid sequences with U+FFFD REPLACEMENT CHARACTER, as is
//! conventionally done in other Rust APIs that deal with string
//! encodings.
//!
//! # `OsStringExt` and `OsStrExt`
//!
//! [`OsString`] is the Rust wrapper for owned strings in the
//! preferred representation of the operating system.  On Windows,
//! this struct gets augmented with an implementation of the
//! [`OsStringExt`] trait, which has a [`from_wide`] method.  This
//! lets you create an [`OsString`] from a `&[u16]` slice; presumably
//! you get such a slice out of a `WCHAR` Windows API.
//!
//! Similarly, [`OsStr`] is the Rust wrapper for borrowed strings from
//! preferred representation of the operating system.  On Windows, the
//! [`OsStrExt`] trait provides the [`encode_wide`] method, which
//! outputs an [`EncodeWide`] iterator.  You can [`collect`] this
//! iterator, for example, to obtain a `Vec<u16>`; you can later get a
//! pointer to this vector's contents and feed it to Windows APIs.
//!
//! These traits, along with [`OsString`] and [`OsStr`], work in
//! conjunction so that it is possible to **round-trip** strings from
//! Windows and back, with no loss of data, even if the strings are
//! ill-formed UTF-16.
//!
//! [ill-formed-utf-16]: https://simonsapin.github.io/wtf-8/#ill-formed-utf-16
//! [`OsString`]: ../../../ffi/struct.OsString.html
//! [`OsStr`]: ../../../ffi/struct.OsStr.html
//! [`OsStringExt`]: trait.OsStringExt.html
//! [`OsStrExt`]: trait.OsStrExt.html
//! [`EncodeWide`]: struct.EncodeWide.html
//! [`from_wide`]: trait.OsStringExt.html#tymethod.from_wide
//! [`encode_wide`]: trait.OsStrExt.html#tymethod.encode_wide
//! [`collect`]: ../../../iter/trait.Iterator.html#method.collect

#![stable(feature = "rust1", since = "1.0.0")]

use ffi::{OsString, OsStr};
use sys::os_str::Buf;
use sys_common::wtf8::Wtf8Buf;
use sys_common::{FromInner, AsInner};

#[stable(feature = "rust1", since = "1.0.0")]
pub use sys_common::wtf8::EncodeWide;

/// Windows-specific extensions to [`ffi::OsString`].
///
/// [`ffi::OsString`]: ../../../../std/ffi/struct.OsString.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStringExt {
    /// Creates an `OsString` from a potentially ill-formed UTF-16 slice of
    /// 16-bit code units.
    ///
    /// This is lossless: calling [`encode_wide`] on the resulting string
    /// will always return the original code units.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    /// ```
    ///
    /// [`encode_wide`]: ./trait.OsStrExt.html#tymethod.encode_wide
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_wide(wide: &[u16]) -> Self;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStringExt for OsString {
    fn from_wide(wide: &[u16]) -> OsString {
        FromInner::from_inner(Buf { inner: Wtf8Buf::from_wide(wide) })
    }
}

/// Windows-specific extensions to [`ffi::OsStr`].
///
/// [`ffi::OsStr`]: ../../../../std/ffi/struct.OsStr.html
#[stable(feature = "rust1", since = "1.0.0")]
pub trait OsStrExt {
    /// Re-encodes an `OsStr` as a wide character sequence, i.e. potentially
    /// ill-formed UTF-16.
    ///
    /// This is lossless: calling [`OsString::from_wide`] and then
    /// `encode_wide` on the result will yield the original code units.
    /// Note that the encoding does not add a final null terminator.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::ffi::OsString;
    /// use std::os::windows::prelude::*;
    ///
    /// // UTF-16 encoding for "Unicode".
    /// let source = [0x0055, 0x006E, 0x0069, 0x0063, 0x006F, 0x0064, 0x0065];
    ///
    /// let string = OsString::from_wide(&source[..]);
    ///
    /// let result: Vec<u16> = string.encode_wide().collect();
    /// assert_eq!(&source[..], &result[..]);
    /// ```
    ///
    /// [`OsString::from_wide`]: ./trait.OsStringExt.html#tymethod.from_wide
    #[stable(feature = "rust1", since = "1.0.0")]
    fn encode_wide(&self) -> EncodeWide;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl OsStrExt for OsStr {
    fn encode_wide(&self) -> EncodeWide {
        self.as_inner().inner.encode_wide()
    }
}
