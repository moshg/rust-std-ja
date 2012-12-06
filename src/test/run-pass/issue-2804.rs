extern mod std;
use io::WriterUtil;
use std::map::HashMap;
use std::json;

enum object
{
    bool_value(bool),
    int_value(i64),
}

fn lookup(table: ~json::Object, key: ~str, default: ~str) -> ~str
{
    match table.find(&key)
    {
        option::Some(std::json::String(copy s)) =>
        {
            copy s
        }
        option::Some(value) =>
        {
            error!("%s was expected to be a string but is a %?", key, value);
            default
        }
        option::None =>
        {
            default
        }
    }
}

fn add_interface(store: int, managed_ip: ~str, data: std::json::Json) -> (~str, object)
{
    match &data
    {
        &std::json::Object(copy interface) =>
        {
            let name = lookup(copy interface, ~"ifDescr", ~"");
            let label = fmt!("%s-%s", managed_ip, name);

            (label, bool_value(false))
        }
        _ =>
        {
            error!("Expected dict for %s interfaces but found %?", managed_ip, data);
            (~"gnos:missing-interface", bool_value(true))
        }
    }
}

fn add_interfaces(store: int, managed_ip: ~str, device: std::map::HashMap<~str, std::json::Json>) -> ~[(~str, object)]
{
    match device[~"interfaces"]
    {
        std::json::List(interfaces) =>
        {
          do vec::map(interfaces) |interface| {
                add_interface(store, managed_ip, *interface)
          }
        }
        _ =>
        {
            error!("Expected list for %s interfaces but found %?", managed_ip, device[~"interfaces"]);
            ~[]
        }
    }
}

fn main() {}
