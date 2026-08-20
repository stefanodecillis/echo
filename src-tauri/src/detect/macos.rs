//! CoreAudio probe for the microphone signal: is something *other than Echo*
//! listening right now?
//!
//! Two things make this hard, and both of them have bitten real people.
//!
//! **1. Telling Echo apart from the meeting.** The obvious property,
//! `kAudioDevicePropertyDeviceIsRunningSomewhere`, stays true as long as *any*
//! client has the default input open — including Echo's own capture. Read on
//! its own it says "something is listening" during every recording Echo makes,
//! which is useless precisely when the watcher needs it most: the auto-stop
//! safety net asks "has the room gone quiet?" and would always hear Echo
//! itself.
//!
//! **2. Telling listening apart from playing.** This is the one people
//! reported: *"it detects a meeting every time there is a sound from any app."*
//!
//! Two separate things say "yes" to the wrong question, and both were live:
//!
//! * **A duplex device.** AirPods, and every other headset, are a speaker and a
//!   microphone behind a single `AudioDeviceID`. Ask that device whether it is
//!   "running somewhere" while music plays and the honest answer is yes, because
//!   the *output* half is running. The header says as much: the property is a
//!   `UInt32` that means "the AudioDevice is running in at least one process",
//!   with no direction in it at all. Nothing in `AudioHardware.h` promises that
//!   narrowing the address to `kAudioObjectPropertyScopeInput` narrows *this*
//!   selector's answer, so the scope alone cannot be trusted to carry the fix —
//!   it is asked for anyway (a driver that does honour it gives a better answer)
//!   and then corroborated with something documented to be direction-specific.
//! * **"Hey Siri".** Worse, and the likelier culprit: `corespeechd` reports
//!   `kAudioProcessPropertyIsRunningInput = 1` permanently, while using no input
//!   device at all — the always-listening trigger does not hold a device the way
//!   an app does. On any Mac with the voice trigger switched on, the per-process
//!   probe therefore said "something else is listening" on *every* poll, for
//!   ever. That is not a false positive now and then; it is a signal jammed on,
//!   and it turns the rest of the heuristic into a coin toss about how long any
//!   other app has been open. Measured, not guessed: the ignored
//!   `what_this_machine_says` test in this file prints it.
//!
//! Both are answered by the same corroboration, and it is verified on real
//! hardware: for `afplay` playing a sound, the input-scoped device list is
//! empty and the output-scoped one has one device in it, so the scope is
//! honoured where it is documented to be; for `corespeechd`, both are empty.
//!
//! So the question is asked in the strongest form available, in this order:
//!
//! 1. **Per process** (macOS 14+): `kAudioHardwarePropertyProcessObjectList`,
//!    then `kAudioProcessPropertyIsRunningInput` for each process whose
//!    `kAudioProcessPropertyPID` is not ours. Stricter (Echo excluded) and
//!    wider (any input device, not only the default one) than anything
//!    device-wide. That property is itself worded around active streams
//!    rather than intent, so it is corroborated with
//!    `kAudioProcessPropertyDevices` **in the input scope** — a property whose
//!    header explicitly says "the scope will select the input or output device
//!    list". A process playing music through a headset uses that headset for
//!    output and has an empty *input* device list, and that is what separates
//!    the two cases.
//! 2. **Per device** (macOS 13, where the process list does not exist): the
//!    default input device's running flag — asked in the input scope, falling
//!    back to the global one where the scope is not published — corroborated by
//!    at least one **active input stream**
//!    (`kAudioDevicePropertyStreams` in the input scope, each stream checked
//!    with `kAudioStreamPropertyDirection` and `kAudioStreamPropertyIsActive`).
//!    On macOS 13 this branch still cannot tell Echo apart from the meeting, so
//!    the auto-stop suggestion stays best-effort there; detection itself is
//!    unaffected, since Echo holds nothing while idle.
//!
//! In both cases a corroborating question that cannot be *asked* leaves the
//! primary answer standing (see [`process_counts_as_input`]). Trading a false
//! positive for a signal that never fires would be the worse bug: a meeting in
//! a browser tab has nothing else to be noticed by.
//!
//! The FFI stays a handful of small functions, wrapped so nothing outside this
//! file touches `coreaudio-sys` directly, and nothing here opens or holds a
//! device: we only ever read properties and let them go (mantra 1). Anything
//! unexpected from a single process, device or stream reads as "cannot tell
//! about that one" and is skipped, never an error for the whole poll.

use std::mem;
use std::os::raw::c_void;

use coreaudio_sys::{
    kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioDevicePropertyStreams,
    kAudioHardwarePropertyDefaultInputDevice, kAudioHardwarePropertyProcessObjectList,
    kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyScopeInput, kAudioObjectSystemObject,
    kAudioProcessPropertyDevices, kAudioProcessPropertyIsRunningInput, kAudioProcessPropertyPID,
    kAudioStreamPropertyDirection, kAudioStreamPropertyIsActive, AudioDeviceID,
    AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize, AudioObjectHasProperty,
    AudioObjectID, AudioObjectPropertyAddress,
};

use super::DetectError;

/// `kAudioObjectPropertyElementMain` (nee `...Master`) is always `0`; spelled
/// out as a literal so this keeps compiling across SDKs that rename it.
const ELEMENT_MAIN: u32 = 0;
const NO_DEVICE: AudioDeviceID = 0;

/// `kAudioStreamPropertyDirection`: "a value of 0 means that this AudioStream
/// is an output stream and a value of 1 means that it is an input stream".
const DIRECTION_INPUT: u32 = 1;

fn read_property<T: Copy>(
    object_id: AudioObjectID,
    address: &AudioObjectPropertyAddress,
    out: &mut T,
) -> Result<(), DetectError> {
    let mut size = mem::size_of::<T>() as u32;
    // SAFETY: `address` is a valid, initialized `AudioObjectPropertyAddress`;
    // `out` is a valid `T`-sized buffer we own for the duration of the call;
    // `size` matches that buffer exactly. CoreAudio writes at most `size`
    // bytes into it and reports how much it actually wrote.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            address as *const AudioObjectPropertyAddress,
            0,
            std::ptr::null(),
            &mut size,
            out as *mut T as *mut c_void,
        )
    };
    if status != 0 {
        return Err(DetectError::Probe(format!(
            "CoreAudio property read failed (status {status})"
        )));
    }
    Ok(())
}

fn default_input_device() -> Result<AudioDeviceID, DetectError> {
    let address = global_address(kAudioHardwarePropertyDefaultInputDevice);
    let mut device_id: AudioDeviceID = NO_DEVICE;
    read_property(kAudioObjectSystemObject, &address, &mut device_id)?;
    Ok(device_id)
}

fn address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: ELEMENT_MAIN,
    }
}

fn global_address(selector: u32) -> AudioObjectPropertyAddress {
    address(selector, kAudioObjectPropertyScopeGlobal)
}

/// The same question, asked about the input half of the object only. Load-bearing
/// for the properties whose headers say the scope selects the direction
/// (`kAudioProcessPropertyDevices`, `kAudioDevicePropertyStreams`).
fn input_address(selector: u32) -> AudioObjectPropertyAddress {
    address(selector, kAudioObjectPropertyScopeInput)
}

fn has_property(object_id: AudioObjectID, address: &AudioObjectPropertyAddress) -> bool {
    // SAFETY: same object/address contract as `read_property`;
    // `AudioObjectHasProperty` only reads `object_id`/`address`, never writes
    // through either pointer.
    unsafe { AudioObjectHasProperty(object_id, address) != 0 }
}

/// Read a list-valued property — the process list, a process's devices, a
/// device's streams.
///
/// `None` means the question could not be asked or answered on this object;
/// `Some(list)` is the answer, and an empty list is a real answer, not a
/// failure. Keeping those apart is the whole point: "this process uses no
/// device for input" is evidence, "I could not find out" is not.
fn read_object_list(
    object_id: AudioObjectID,
    address: &AudioObjectPropertyAddress,
) -> Option<Vec<AudioObjectID>> {
    if !has_property(object_id, address) {
        return None;
    }

    let mut bytes: u32 = 0;
    // SAFETY: `address` is a valid, initialized address; `bytes` is a valid
    // `UInt32` we own. The call only writes the required size into it.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(object_id, address, 0, std::ptr::null(), &mut bytes)
    };
    if status != 0 {
        return None;
    }

    let capacity = bytes as usize / mem::size_of::<AudioObjectID>();
    if capacity == 0 {
        return Some(Vec::new());
    }
    let mut objects: Vec<AudioObjectID> = vec![0; capacity];
    let mut bytes_written = bytes;
    // SAFETY: `objects` holds exactly `bytes_written` bytes of `AudioObjectID`
    // storage we own; CoreAudio writes at most that many and reports how many it
    // actually wrote.
    let status = unsafe {
        AudioObjectGetPropertyData(
            object_id,
            address,
            0,
            std::ptr::null(),
            &mut bytes_written,
            objects.as_mut_ptr().cast::<c_void>(),
        )
    };
    if status != 0 {
        return None;
    }
    objects.truncate(bytes_written as usize / mem::size_of::<AudioObjectID>());
    Some(objects)
}

/// Does one process's pair of answers add up to "this process is listening"?
///
/// `running_input` is `kAudioProcessPropertyIsRunningInput`, which the header
/// defines around *active streams* — "the process is running IO and there is at
/// least one active input stream" — so on a duplex device (any headset) it can
/// be true of a process that is only playing sound.
/// `input_devices` is how many devices that process uses **for input**, from the
/// input-scoped `kAudioProcessPropertyDevices`, or `None` when that could not be
/// read at all.
///
/// A process playing music through AirPods uses them for output and nothing for
/// input: `Some(0)`, and no meeting. `corespeechd`, waiting for "Hey Siri" with
/// the flag stuck on for ever, is also `Some(0)`. A process that will not answer
/// the second question at all keeps the benefit of the doubt, because a mic
/// signal that never fires would take browser meetings — which have no other
/// signal at all — with it.
fn process_counts_as_input(running_input: bool, input_devices: Option<usize>) -> bool {
    if !running_input {
        return false;
    }
    // Only an explicit, successful "no input devices" overrules the flag.
    input_devices != Some(0)
}

/// The same shape of judgement for the macOS 13 fallback: the device is running
/// for somebody, and it has an input stream that is actually doing IO.
///
/// `active_input_streams` is `None` when the device does not publish its streams
/// under the input scope — in which case the running flag stands on its own, as
/// it did before, rather than turning into a signal that can never fire.
fn device_counts_as_input(running_somewhere: bool, active_input_streams: Option<usize>) -> bool {
    if !running_somewhere {
        return false;
    }
    active_input_streams != Some(0)
}

/// How many devices this process is using **for input**.
///
/// `None` when the input-scoped device list cannot be read — see
/// [`process_counts_as_input`] for what that is allowed to mean.
fn process_input_device_count(process: AudioObjectID) -> Option<usize> {
    read_object_list(process, &input_address(kAudioProcessPropertyDevices)).map(|d| d.len())
}

/// Is this stream an input stream that is doing IO right now?
fn stream_is_active_input(stream: AudioObjectID) -> bool {
    let mut direction: u32 = 0;
    let direction_address = global_address(kAudioStreamPropertyDirection);
    if read_property(stream, &direction_address, &mut direction).is_ok()
        && direction != DIRECTION_INPUT
    {
        // Belt and braces: the input scope should have filtered this out
        // already, and a device that lists an output stream under the input
        // scope is exactly the confusion this file exists to avoid.
        return false;
    }

    let mut active: u32 = 0;
    let active_address = global_address(kAudioStreamPropertyIsActive);
    read_property(stream, &active_address, &mut active)
        .map(|()| active != 0)
        .unwrap_or(false)
}

/// How many of this device's input streams are enabled and doing IO, or `None`
/// when the device does not publish input streams at all.
fn active_input_stream_count(device_id: AudioDeviceID) -> Option<usize> {
    let streams = read_object_list(device_id, &input_address(kAudioDevicePropertyStreams))?;
    Some(
        streams
            .into_iter()
            .filter(|stream| stream_is_active_input(*stream))
            .count(),
    )
}

/// Is any process other than this one running input right now?
///
/// `Ok(None)` means the question cannot be asked on this macOS (the
/// per-process object list arrived in macOS 14) — the caller falls back to the
/// device-wide flag.
fn another_process_is_running_input() -> Result<Option<bool>, DetectError> {
    let list_address = global_address(kAudioHardwarePropertyProcessObjectList);
    let Some(processes) = read_object_list(kAudioObjectSystemObject, &list_address) else {
        return Ok(None);
    };

    let ours = std::process::id() as i32;
    for process in processes {
        let mut running: u32 = 0;
        let running_address = global_address(kAudioProcessPropertyIsRunningInput);
        if !has_property(process, &running_address)
            || read_property(process, &running_address, &mut running).is_err()
            || running == 0
        {
            continue;
        }

        let mut pid: i32 = 0;
        let pid_address = global_address(kAudioProcessPropertyPID);
        if read_property(process, &pid_address, &mut pid).is_ok() && pid == ours {
            // That is Echo's own capture. Not evidence of anything.
            continue;
        }

        // Running IO on a device that happens to have a microphone in it is not
        // the same as using the microphone. Ask which devices this process uses
        // *for input* before believing it.
        if process_counts_as_input(true, process_input_device_count(process)) {
            return Ok(Some(true));
        }
    }
    Ok(Some(false))
}

/// The macOS 13 fallback: what the default input device says about itself.
///
/// Counts Echo's own capture too — there is no way to separate the two here —
/// and needs an active input stream behind the running flag, so playback
/// through a headset is not mistaken for someone talking.
fn default_input_device_is_running() -> Result<bool, DetectError> {
    let device_id = default_input_device()?;
    if device_id == NO_DEVICE {
        return Ok(false);
    }

    // Ask about the input half first. The selector is documented as
    // device-wide, so this is an improvement where a driver honours it and a
    // no-op where it does not; the stream check below is what actually carries
    // the direction.
    let scoped = input_address(kAudioDevicePropertyDeviceIsRunningSomewhere);
    let global = global_address(kAudioDevicePropertyDeviceIsRunningSomewhere);
    let address = if has_property(device_id, &scoped) {
        scoped
    } else if has_property(device_id, &global) {
        global
    } else {
        // Neither scope publishes it: nothing is running on it that we can see.
        return Ok(false);
    };

    let mut running: u32 = 0;
    read_property(device_id, &address, &mut running)?;
    Ok(device_counts_as_input(
        running != 0,
        active_input_stream_count(device_id),
    ))
}

/// Is some app other than Echo already using an input device?
///
/// `Ok(false)` — not an error — when there is no default input device at
/// all, or the property does not apply to it; both just mean there is
/// nothing running on it right now.
pub fn input_device_in_use() -> Result<bool, DetectError> {
    match another_process_is_running_input() {
        Ok(Some(answer)) => return Ok(answer),
        Ok(None) => {
            tracing::debug!("no per-process audio view on this macOS; asking the device instead");
        }
        Err(error) => {
            tracing::debug!(%error, "could not ask which processes are listening");
        }
    }

    default_input_device_is_running()
}

#[cfg(test)]
mod tests {
    use super::*;

    // These cover the judgement, which is where the field report lived: the FFI
    // itself needs a duplex headset and two macOS versions to exercise, and is
    // deliberately kept to reads that cannot fail into a "yes".

    #[test]
    fn a_flag_with_no_input_device_behind_it_is_not_the_microphone() {
        // Both halves of the reported bug read exactly like this: a process with
        // the running-input flag set that uses no device for input at all —
        // music through a duplex headset, and `corespeechd` waiting for
        // "Hey Siri" with the flag stuck on for ever.
        assert!(!process_counts_as_input(true, Some(0)));
    }

    #[test]
    fn a_process_actually_recording_counts() {
        assert!(process_counts_as_input(true, Some(1)));
    }

    #[test]
    fn a_process_running_nothing_never_counts() {
        assert!(!process_counts_as_input(false, Some(0)));
        assert!(!process_counts_as_input(false, Some(2)));
        assert!(!process_counts_as_input(false, None));
    }

    #[test]
    fn a_question_that_cannot_be_asked_leaves_the_flag_standing() {
        // Better a signal that is occasionally too generous than one that never
        // fires: a meeting in a browser tab has nothing else to be noticed by.
        assert!(process_counts_as_input(true, None));
    }

    #[test]
    fn a_device_needs_an_input_stream_behind_its_running_flag() {
        // Headset playing music on macOS 13: running, no active input stream.
        assert!(!device_counts_as_input(true, Some(0)));
        assert!(device_counts_as_input(true, Some(1)));
        assert!(!device_counts_as_input(false, Some(3)));
        // Streams not published: the flag stands, as it always did.
        assert!(device_counts_as_input(true, None));
        assert!(!device_counts_as_input(false, None));
    }

    #[test]
    fn the_input_scope_is_a_different_question_from_the_global_one() {
        let global = global_address(kAudioDevicePropertyStreams);
        let input = input_address(kAudioDevicePropertyStreams);
        assert_eq!(global.mSelector, input.mSelector);
        assert_ne!(
            global.mScope, input.mScope,
            "asking about the input half has to reach CoreAudio as a \
             different address, or the fix is a comment"
        );
        assert_eq!(input.mScope, kAudioObjectPropertyScopeInput);
        assert_eq!(input.mElement, ELEMENT_MAIN);
    }

    /// Not a test: a way to look at what this machine's CoreAudio actually says,
    /// which is the only way to check the FFI half of this file. Run it with
    /// something recording and then with nothing recording:
    ///
    /// ```text
    /// cargo test detect::macos::tests::what_this_machine_says -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "diagnostic; prints what this machine's CoreAudio reports"]
    fn what_this_machine_says() {
        let list_address = global_address(kAudioHardwarePropertyProcessObjectList);
        let processes = read_object_list(kAudioObjectSystemObject, &list_address);
        println!("process objects: {:?}", processes.as_ref().map(|p| p.len()));
        for process in processes.into_iter().flatten() {
            let mut running: u32 = 0;
            let running_address = global_address(kAudioProcessPropertyIsRunningInput);
            let has = has_property(process, &running_address);
            let read = read_property(process, &running_address, &mut running);
            let mut pid: i32 = 0;
            let _ = read_property(process, &global_address(kAudioProcessPropertyPID), &mut pid);
            let inputs = process_input_device_count(process);
            let mut out_running: u32 = 0;
            let _ = read_property(
                process,
                &global_address(coreaudio_sys::kAudioProcessPropertyIsRunningOutput),
                &mut out_running,
            );
            let outputs = read_object_list(
                process,
                &address(
                    kAudioProcessPropertyDevices,
                    coreaudio_sys::kAudioObjectPropertyScopeOutput,
                ),
            )
            .map(|d| d.len());
            let all = read_object_list(process, &global_address(kAudioProcessPropertyDevices))
                .map(|d| d.len());
            if running != 0 || out_running != 0 || inputs.unwrap_or(0) > 0 {
                println!(
                    "  pid {pid}: running_input={running} running_output={out_running} \
                     (has={has}, read_ok={}) input_devices={inputs:?} \
                     output_devices={outputs:?} all_devices={all:?} -> counts={}",
                    read.is_ok(),
                    process_counts_as_input(running != 0, inputs)
                );
            }
        }
        let device = default_input_device();
        println!("default input device: {device:?}");
        if let Ok(device_id) = device {
            println!(
                "  active input streams: {:?}",
                active_input_stream_count(device_id)
            );
            println!(
                "  device says running: {:?}",
                default_input_device_is_running()
            );
        }
        println!("input_device_in_use() -> {:?}", input_device_in_use());
    }

    #[test]
    fn the_stream_direction_constant_matches_the_header() {
        // AudioHardwareBase.h: 0 is an output stream, 1 is an input stream. Get
        // this backwards and the fallback believes exactly the wrong half.
        assert_eq!(DIRECTION_INPUT, 1);
    }
}
