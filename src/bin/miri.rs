#![feature(rustc_private)]

extern crate getopts;
extern crate miri;
extern crate rustc;
extern crate rustc_driver;
extern crate env_logger;
extern crate log_settings;
extern crate syntax;
#[macro_use] extern crate log;

use miri::eval_main;
use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls, Compilation};

struct MiriCompilerCalls;

impl<'a> CompilerCalls<'a> for MiriCompilerCalls {
    fn build_controller(
        &mut self,
        _: &Session,
        _: &getopts::Matches
    ) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();

        control.after_analysis.stop = Compilation::Stop;
        control.after_analysis.callback = Box::new(|state| {
            state.session.abort_if_errors();

            let tcx = state.tcx.unwrap();
            let mir_map = state.mir_map.unwrap();
            let (node_id, _) = state.session.entry_fn.borrow()
                .expect("no main or start function found");
            eval_main(tcx, mir_map, node_id);

            state.session.abort_if_errors();
        });

        control
    }
}

fn init_logger() {
    const MAX_INDENT: usize = 40;

    let format = |record: &log::LogRecord| {
        if record.level() == log::LogLevel::Trace {
            // prepend spaces to indent the final string
            let indentation = log_settings::settings().indentation;
            format!("{lvl}:{module}{depth:2}{indent:<indentation$} {text}",
                lvl = record.level(),
                module = record.location().module_path(),
                depth = indentation / MAX_INDENT,
                indentation = indentation % MAX_INDENT,
                indent = "",
                text = record.args())
        } else {
            format!("{lvl}:{module}: {text}",
                lvl = record.level(),
                module = record.location().module_path(),
                text = record.args())
        }
    };

    let mut builder = env_logger::LogBuilder::new();
    builder.format(format).filter(None, log::LogLevelFilter::Info);

    if std::env::var("MIRI_LOG").is_ok() {
        builder.parse(&std::env::var("MIRI_LOG").unwrap());
    }

    builder.init().unwrap();
}

fn find_sysroot() -> String {
    // Taken from https://github.com/Manishearth/rust-clippy/pull/911.
    let home = option_env!("RUSTUP_HOME").or(option_env!("MULTIRUST_HOME"));
    let toolchain = option_env!("RUSTUP_TOOLCHAIN").or(option_env!("MULTIRUST_TOOLCHAIN"));
    match (home, toolchain) {
        (Some(home), Some(toolchain)) => format!("{}/toolchains/{}", home, toolchain),
        _ => option_env!("RUST_SYSROOT")
            .expect("need to specify RUST_SYSROOT env var or use rustup or multirust")
            .to_owned(),
    }
}

fn main() {
    init_logger();
    let mut args: Vec<String> = std::env::args().collect();

    let sysroot_flag = String::from("--sysroot");
    if !args.contains(&sysroot_flag) {
        args.push(sysroot_flag);
        args.push(find_sysroot());
    }

    rustc_driver::run_compiler(&args, &mut MiriCompilerCalls);
}
