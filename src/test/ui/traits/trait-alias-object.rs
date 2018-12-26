#![feature(trait_alias)]

trait EqAlias = Eq;
trait IteratorAlias = Iterator;

fn main() {
    let _: &dyn EqAlias = &123; //~ ERROR `EqAlias` cannot be made into an object
    let _: &dyn IteratorAlias = &vec![123].into_iter(); //~ ERROR must be specified
}
