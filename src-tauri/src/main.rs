// Echo's desktop entry point. Everything lives in the library so tests and the
// mobile/CLI shells can share it.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    echo_lib::run()
}
