use libc::{c_char, c_int};

#[link_args = "-Llinenoise"]
#[link_name = "linenoise"]
#[abi = "cdecl"]
extern mod linenoise {
    #[legacy_exports];
    fn linenoise(prompt: *c_char) -> *c_char;
    fn linenoiseHistoryAdd(line: *c_char) -> c_int;
    fn linenoiseHistorySetMaxLen(len: c_int) -> c_int;
    fn linenoiseHistorySave(file: *c_char) -> c_int;
    fn linenoiseHistoryLoad(file: *c_char) -> c_int;
    fn linenoiseSetCompletionCallback(callback: *u8);
    fn linenoiseAddCompletion(completions: *(), line: *c_char);
    fn linenoiseClearScreen();
}

/// Add a line to history
pub fn add_history(line: ~str) -> bool {
	do str::as_c_str(line) |buf| {
		linenoise::linenoiseHistoryAdd(buf) == 1 as c_int
	}
}

/// Set the maximum amount of lines stored
pub fn set_history_max_len(len: int) -> bool {
	linenoise::linenoiseHistorySetMaxLen(len as c_int) == 1 as c_int
}

/// Save line history to a file
pub fn save_history(file: ~str) -> bool {
	do str::as_c_str(file) |buf| {
		linenoise::linenoiseHistorySave(buf) == 1 as c_int
	}
}

/// Load line history from a file
pub fn load_history(file: ~str) -> bool {
	do str::as_c_str(file) |buf| {
		linenoise::linenoiseHistoryLoad(buf) == 1 as c_int
	}
}

/// Print out a prompt and then wait for input and return it
pub fn read(prompt: ~str) -> Option<~str> {
	do str::as_c_str(prompt) |buf| unsafe {
		let line = linenoise::linenoise(buf);

		if line.is_null() { None }
		else { Some(str::raw::from_c_str(line)) }
	}
}

/// Clear the screen
pub fn clear() {
	linenoise::linenoiseClearScreen();
}

pub type CompletionCb = fn~(~str, fn(~str));

fn complete_key(_v: @CompletionCb) {}

/// Bind to the main completion callback
pub fn complete(cb: CompletionCb) unsafe {
	task::local_data::local_data_set(complete_key, @(move cb));

	extern fn callback(line: *c_char, completions: *()) unsafe {
		let cb: CompletionCb = copy *task::local_data::local_data_get(complete_key).get();

		do cb(str::raw::from_c_str(line)) |suggestion| {
			do str::as_c_str(suggestion) |buf| {
				linenoise::linenoiseAddCompletion(completions, buf);
			}
		}
	}

	linenoise::linenoiseSetCompletionCallback(callback);
}
