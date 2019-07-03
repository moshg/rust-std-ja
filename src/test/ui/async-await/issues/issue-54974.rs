// compile-pass
// edition:2018

#![feature(async_await)]

use std::sync::Arc;

trait SomeTrait: Send + Sync + 'static {
    fn do_something(&self);
}

async fn my_task(obj: Arc<dyn SomeTrait>) {
    unimplemented!()
}

fn main() {}
