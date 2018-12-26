#![feature(staged_api, allow_internal_unstable)]
#![stable(feature = "stable", since = "1.0.0")]

#[unstable(feature = "function", issue = "0")]
pub fn unstable() {}


#[stable(feature = "stable", since = "1.0.0")]
pub struct Foo {
    #[unstable(feature = "struct_field", issue = "0")]
    pub x: u8
}

impl Foo {
    #[unstable(feature = "method", issue = "0")]
    pub fn method(&self) {}
}

#[stable(feature = "stable", since = "1.0.0")]
pub struct Bar {
    #[unstable(feature = "struct2_field", issue = "0")]
    pub x: u8
}

#[stable(feature = "stable", since = "1.0.0")]
#[allow_internal_unstable]
#[macro_export]
macro_rules! call_unstable_allow {
    () => { $crate::unstable() }
}

#[stable(feature = "stable", since = "1.0.0")]
#[allow_internal_unstable]
#[macro_export]
macro_rules! construct_unstable_allow {
    ($e: expr) => {
        $crate::Foo { x: $e }
    }
}

#[stable(feature = "stable", since = "1.0.0")]
#[allow_internal_unstable]
#[macro_export]
macro_rules! call_method_allow {
    ($e: expr) => { $e.method() }
}

#[stable(feature = "stable", since = "1.0.0")]
#[allow_internal_unstable]
#[macro_export]
macro_rules! access_field_allow {
    ($e: expr) => { $e.x }
}

#[stable(feature = "stable", since = "1.0.0")]
#[allow_internal_unstable]
#[macro_export]
macro_rules! pass_through_allow {
    ($e: expr) => { $e }
}

#[stable(feature = "stable", since = "1.0.0")]
#[macro_export]
macro_rules! call_unstable_noallow {
    () => { $crate::unstable() }
}

#[stable(feature = "stable", since = "1.0.0")]
#[macro_export]
macro_rules! construct_unstable_noallow {
    ($e: expr) => {
        $crate::Foo { x: $e }
    }
}

#[stable(feature = "stable", since = "1.0.0")]
#[macro_export]
macro_rules! call_method_noallow {
    ($e: expr) => { $e.method() }
}

#[stable(feature = "stable", since = "1.0.0")]
#[macro_export]
macro_rules! access_field_noallow {
    ($e: expr) => { $e.x }
}

#[stable(feature = "stable", since = "1.0.0")]
#[macro_export]
macro_rules! pass_through_noallow {
    ($e: expr) => { $e }
}
