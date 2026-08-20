//! CoreAudio probe for the microphone signal: is something *other than Echo*
//! listening right now?
//!
//! "Other than Echo" is the whole difficulty. The obvious property,
//! `kAudioDevicePropertyDeviceIsRunningSomewhere`, stays true as long as *any*
//! client has the default input open — including Echo's own capture. Read on
//! its own it says "something is listening" during every recording Echo makes,
//! which is useless precisely when the watcher needs it most: the auto-stop
//! safety net asks "has the room gone quiet?" and would always hear Echo
//! itself.
//!
//! So this file asks the per-process question first
//! (`kAudioHardwarePropertyProcessObjectList` plus
//! `kAudioProcessPropertyIsRunningInput`, macOS 14+): is any process *whose PID
//! is not ours* running input? That is both stricter (Echo excluded) and wider
//! (any input device, not only the default one) than the device-wide flag.
//! Where that view does not exist — macOS 13 — it falls back to the device-wide
//! flag and says so in the docs: on macOS 13 the mic signal cannot tell Echo
//! apart from the meeting, so the auto-stop suggestion stays best-effort there.
//! Detection itself is unaffected, since Echo holds nothing while idle.
//!
//! The FFI stays a handful of lines, wrapped so nothing outside this file
//! touches `coreaudio-sys` directly, and nothing here opens or holds a device:
//! we only ever read properties and let them go (mantra 1). Anything unexpected
//! from a single process reads as "cannot tell about that one" and is skipped,
//! never an error for the whole poll.

use std::mem;
use std::os::raw::c_void;

use coreaudio_sys::{
    kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioHardwarePropertyDefaultInputDevice,
    kAudioHardwarePropertyProcessObjectList, kAudioObjectPropertyScopeGlobal,
    kAudioObjectSystemObject, kAudioProcessPropertyIsRunningInput, kAudioProcessPropertyPID,
    AudioDeviceID, AudioObjectGetPropertyData, AudioObjectGetPropertyDataSize,
    AudioObjectHasProperty, AudioObjectID, AudioObjectPropertyAddress,
};

use super::DetectError;

/// `kAudioObjectPropertyElementMain` (nee `...Master`) is always `0`; spelled
/// out as a literal so this keeps compiling across SDKs that rename it.
const ELEMENT_MAIN: u32 = 0;
const NO_DEVICE: AudioDeviceID = 0;

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

fn global_address(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: ELEMENT_MAIN,
    }
}

fn has_property(object_id: AudioObjectID, address: &AudioObjectPropertyAddress) -> bool {
    // SAFETY: same object/address contract as `read_property`;
    // `AudioObjectHasProperty` only reads `object_id`/`address`, never writes
    // through either pointer.
    unsafe { AudioObjectHasProperty(object_id, address) != 0 }
}

/// Is any process other than this one running input right now?
///
/// `Ok(None)` means the question cannot be asked on this macOS (the
/// per-process object list arrived in macOS 14) — the caller falls back to the
/// device-wide flag.
fn another_process_is_running_input() -> Result<Option<bool>, DetectError> {
    let address = global_address(kAudioHardwarePropertyProcessObjectList);
    if !has_property(kAudioObjectSystemObject, &address) {
        return Ok(None);
    }

    let mut bytes: u32 = 0;
    // SAFETY: `address` is a valid, initialized address; `bytes` is a valid
    // `UInt32` we own. The call only writes the required size into it.
    let status = unsafe {
        AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut bytes,
        )
    };
    if status != 0 {
        return Ok(None);
    }

    let capacity = bytes as usize / mem::size_of::<AudioObjectID>();
    if capacity == 0 {
        return Ok(Some(false));
    }
    let mut processes: Vec<AudioObjectID> = vec![0; capacity];
    let mut bytes_written = bytes;
    // SAFETY: `processes` holds exactly `bytes_written` bytes of `AudioObjectID`
    // storage we own; CoreAudio writes at most that many and reports how many it
    // actually wrote.
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &address,
            0,
            std::ptr::null(),
            &mut bytes_written,
            processes.as_mut_ptr().cast::<c_void>(),
        )
    };
    if status != 0 {
        return Ok(None);
    }
    processes.truncate(bytes_written as usize / mem::size_of::<AudioObjectID>());

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
        return Ok(Some(true));
    }
    Ok(Some(false))
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

    let device_id = default_input_device()?;
    if device_id == NO_DEVICE {
        return Ok(false);
    }

    let address = global_address(kAudioDevicePropertyDeviceIsRunningSomewhere);
    if !has_property(device_id, &address) {
        return Ok(false);
    }

    // Note: this branch counts Echo's own capture too — there is no way to
    // separate the two here. See the module docs.
    let mut running: u32 = 0;
    read_property(device_id, &address, &mut running)?;
    Ok(running != 0)
}
