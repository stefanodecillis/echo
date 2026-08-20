//! PipeWire probe for signal (b) on Linux: is some other app's input stream
//! active right now?
//!
//! UNVERIFIED: this crate only has macOS hardware in CI/dev for the detect
//! module so far (see notes for the integrator). It is written to the
//! `pipewire` crate's ordinary registry-plus-sync pattern and kept isolated
//! behind `cfg(target_os = "linux")` precisely so it cannot affect the
//! macOS build either way; it needs a real Linux box before this signal can
//! be trusted. If anything about it does not compile as-is, the fix is
//! local to this file.
//!
//! Bounded and best-effort by design (mantra 1: this is a nice-to-have
//! signal, never load-bearing): connect, look at the registry for at most
//! [`PROBE_TIMEOUT`], then let everything drop. Any trouble at all — PipeWire
//! not running, a socket hiccup, whatever — reads as "no signal", not an
//! error, so a broken desktop session never breaks the idle poll.

use std::cell::Cell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use pipewire as pw;

use super::DetectError;

const PROBE_TIMEOUT: Duration = Duration::from_millis(200);

/// Is some other app's microphone/input stream active right now?
pub fn input_device_in_use() -> Result<bool, DetectError> {
    // A broken or absent PipeWire session is not this watcher's problem to
    // report; it just means we cannot see this signal right now.
    Ok(probe().unwrap_or(false))
}

fn probe() -> Result<bool, pw::Error> {
    pw::init();

    let main_loop = pw::main_loop::MainLoop::new(None)?;
    let context = pw::context::Context::new(&main_loop)?;
    let core = context.connect(None)?;
    let registry = core.get_registry()?;

    let found = Rc::new(Cell::new(false));
    let found_for_listener = found.clone();

    let _registry_listener = registry
        .add_listener_local()
        .global(move |global| {
            let Some(props) = global.props else {
                return;
            };
            let is_input_stream = props
                .get("media.class")
                .map(|class| class == "Stream/Input/Audio")
                .unwrap_or(false);
            if !is_input_stream {
                return;
            }
            // Echo's own capture, if it happens to be running, tags itself;
            // everyone else counts as "something else is listening". Excluding
            // ourselves is load-bearing, not tidiness: the auto-stop safety net
            // asks whether the room has gone quiet *while Echo is recording it*,
            // and would otherwise only ever hear itself.
            let is_echo = props
                .get("application.name")
                .map(|name| name == "Echo")
                .unwrap_or(false);
            if !is_echo {
                found_for_listener.set(true);
            }
        })
        .register();

    let done = Rc::new(Cell::new(false));
    let done_for_listener = done.clone();
    let _core_listener = core
        .add_listener_local()
        .done(move |_id, _seq| done_for_listener.set(true))
        .register();
    core.sync(0)?;

    // No timer source: pump the loop ourselves so this can never block past
    // `PROBE_TIMEOUT`, whether or not the `done` callback ever arrives.
    let deadline = Instant::now() + PROBE_TIMEOUT;
    while !done.get() && Instant::now() < deadline {
        main_loop.loop_().iterate(Duration::from_millis(20));
    }

    Ok(found.get())
}
