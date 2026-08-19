//! CoreAudio probe for signal (b): is something already using the default
//! input device?
//!
//! `kAudioDevicePropertyDeviceIsRunningSomewhere` stays true as long as *any*
//! client — another app, or Echo itself — has the default input device
//! open. That is exactly "something is listening" (DESIGN §3). The FFI is a
//! handful of lines, wrapped so nothing outside this file touches
//! `coreaudio-sys` directly, and nothing here opens or holds the device: we
//! only ever read one property and let it go (mantra 1).

use std::mem;
use std::os::raw::c_void;

use coreaudio_sys::{
    kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioHardwarePropertyDefaultInputDevice,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, AudioDeviceID,
    AudioObjectGetPropertyData, AudioObjectHasProperty, AudioObjectID, AudioObjectPropertyAddress,
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
    let address = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultInputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: ELEMENT_MAIN,
    };
    let mut device_id: AudioDeviceID = NO_DEVICE;
    read_property(kAudioObjectSystemObject, &address, &mut device_id)?;
    Ok(device_id)
}

/// Is some other app (or Echo itself) already using the default input?
///
/// `Ok(false)` — not an error — when there is no default input device at
/// all, or the property does not apply to it; both just mean there is
/// nothing running on it right now.
pub fn input_device_in_use() -> Result<bool, DetectError> {
    let device_id = default_input_device()?;
    if device_id == NO_DEVICE {
        return Ok(false);
    }

    let address = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyDeviceIsRunningSomewhere,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: ELEMENT_MAIN,
    };

    // SAFETY: same object/address contract as `read_property`; `AudioObjectHasProperty`
    // only reads `device_id`/`address`, never writes through either pointer.
    let has_property = unsafe { AudioObjectHasProperty(device_id, &address) };
    if has_property == 0 {
        return Ok(false);
    }

    let mut running: u32 = 0;
    read_property(device_id, &address, &mut running)?;
    Ok(running != 0)
}
