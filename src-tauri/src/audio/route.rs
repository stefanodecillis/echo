//! Where the computer's audio comes out: a loudspeaker in the room, or
//! something strapped to somebody's ears.
//!
//! This is the gate in front of [`crate::audio::bleed`]. If the far side is
//! playing through the laptop speakers there is a path back into the
//! microphone, and everything the far side says arrives twice. If it is playing
//! into headphones there is no such path, and nothing this machine does can put
//! the far side's voice into the microphone.
//!
//! ## The gate is not the safety property
//!
//! **The predicate is.** [`crate::audio::bleed::is_bleed`] is what decides
//! whether a stretch is deleted, and it is built to refuse rather than to
//! delete. This file exists to skip work that cannot pay, to shrink the blast
//! radius of a mistake, and to make a field report readable — "armed on
//! built-in speakers, examined 74, suppressed 12" is a sentence somebody can
//! check. It is not a second safety net, and nothing here should ever be asked
//! to carry a decision the predicate cannot.
//!
//! That is why **[`Route::CannotTell`] arms.** Three arguments, in order of
//! weight:
//!
//! 1. On headphones the correlator cannot hit above chance, because the sound
//!    the microphone would have to be a copy of never entered the room. Arming
//!    there costs one sub-millisecond correlation per utterance and finds
//!    nothing.
//! 2. Disarming on `CannotTell` would disarm on **most Bluetooth** — and a
//!    Bluetooth speakerphone on a meeting-room table is exactly where the worst
//!    bleed in the world lives. AirPods and that speakerphone are
//!    *indistinguishable* from these two properties: both answer `blue`, and
//!    neither publishes a data source that says which one it is. There is no
//!    cleverness available here, only a choice about which way to be wrong.
//! 3. It is the house rule. [`crate::audio::vad::Utterance::measured_voice_ms`]
//!    answers "nobody measured" with "assume it was speech" rather than "assume
//!    it was silence", and `detect/macos.rs`'s `process_counts_as_input(true,
//!    None)` lets a question that cannot be asked leave the primary answer
//!    standing. "I could not find out" is not evidence, and it may not be
//!    spent as if it were.
//!
//! ## The shape of the file is `detect/macos.rs`'s
//!
//! Small `unsafe` readers at the edges, one **pure** function that decides, and
//! the tests on the pure function. Nothing here opens or holds a device: two
//! property reads and they are let go (mantra 1). Anything unreadable reads as
//! `CannotTell`, never as an error and never as a `Route::Headphones` that
//! would quietly switch bleed detection off.

/// Where the far side is coming out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Route {
    /// Into the room, where a microphone can hear it.
    Loudspeaker,
    /// Into somebody's ears. No acoustic path exists.
    Headphones,
    /// The properties do not say — Bluetooth, USB, a virtual device — or they
    /// could not be read at all. **This arms**; see the module docs.
    #[default]
    CannotTell,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Loudspeaker => "loudspeaker",
            Route::Headphones => "headphones",
            Route::CannotTell => "cannot tell",
        }
    }

    /// Whether bleed detection should run on this route.
    ///
    /// Only headphones say no, and they say it because there is no acoustic
    /// path at all — not because the answer is inconvenient.
    pub fn arms(self) -> bool {
        self != Route::Headphones
    }
}

/// Four characters, big-endian, the way CoreAudio spells every one of these
/// constants — `bltn`, `hdpn`, `ispk`. Written this way so the value in the
/// source is readable as the four letters in Apple's header rather than as a
/// decimal nobody can check.
const fn fourcc(code: [u8; 4]) -> u32 {
    u32::from_be_bytes(code)
}

// The transports. These *are* in the generated `coreaudio-sys` bindings; they
// are spelled out here anyway so that [`verdict`] and its tests are ordinary
// Rust on every platform, and a macOS-only test below pins each one against the
// binding it mirrors. If one of these ever stops matching the header, that test
// fails rather than this file quietly judging the wrong transport.
const TRANSPORT_BUILT_IN: u32 = fourcc(*b"bltn");
const TRANSPORT_HDMI: u32 = fourcc(*b"hdmi");
const TRANSPORT_DISPLAY_PORT: u32 = fourcc(*b"dprt");
const TRANSPORT_AIR_PLAY: u32 = fourcc(*b"airp");

// The data sources on the built-in output. **These are not in the headers** —
// `AudioHardwareBase.h` documents `kAudioDevicePropertyDataSource` as a
// four-character code chosen by the driver and lists none of the built-in
// output's own codes. They are defined from their FourCC bytes, with a test
// pinning the values, exactly as `detect/macos.rs` pins `DIRECTION_INPUT`.
/// The speakers inside the machine.
const DATA_SOURCE_INTERNAL_SPEAKER: u32 = fourcc(*b"ispk");
/// Speakers in the analogue jack — powered desk speakers, a monitor's input.
/// Still a loudspeaker in a room, which is all this file cares about.
const DATA_SOURCE_EXTERNAL_SPEAKER: u32 = fourcc(*b"espk");
/// Headphones in the jack.
const DATA_SOURCE_HEADPHONES: u32 = fourcc(*b"hdpn");

/// The whole judgement, over two numbers CoreAudio was asked for.
///
/// `transport` is `kAudioDevicePropertyTransportType` and `data_source` is
/// `kAudioDevicePropertyDataSource` **in the output scope**; `None` means the
/// property could not be read. The table:
///
/// ```text
///   built-in + ispk / espk        loudspeaker   the machine's own speakers, or
///                                               powered speakers in the jack
///   built-in + hdpn               headphones    the jack is the only place on a
///                                               Mac that says so out loud
///   HDMI / DisplayPort / AirPlay  loudspeaker   a television, a monitor, a
///                                               HomePod: all of them play into
///                                               a room
///   Bluetooth / BluetoothLE       cannot tell   AirPods and a speakerphone are
///   USB / virtual / aggregate                   the same two numbers
///   anything unreadable           cannot tell
/// ```
///
/// The Bluetooth row is the important one and it is a statement of fact rather
/// than a gap to be worked around later: a Bluetooth output device reports the
/// transport `blue` and publishes no data source distinguishing an earpiece
/// from a loudspeaker. The same is true of USB. `CannotTell` arms (module
/// docs), so this costs a correlation and never a suppression.
pub fn verdict(transport: Option<u32>, data_source: Option<u32>) -> Route {
    match transport {
        Some(TRANSPORT_BUILT_IN) => match data_source {
            Some(DATA_SOURCE_INTERNAL_SPEAKER) | Some(DATA_SOURCE_EXTERNAL_SPEAKER) => {
                Route::Loudspeaker
            }
            Some(DATA_SOURCE_HEADPHONES) => Route::Headphones,
            // A built-in output whose data source is something else, or which
            // will not answer at all: the honest answer is that we do not know.
            _ => Route::CannotTell,
        },
        Some(TRANSPORT_HDMI) | Some(TRANSPORT_DISPLAY_PORT) | Some(TRANSPORT_AIR_PLAY) => {
            Route::Loudspeaker
        }
        _ => Route::CannotTell,
    }
}

// ---------------------------------------------------------------------------
// The edges
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
mod ffi {
    use std::mem;
    use std::os::raw::c_void;

    use coreaudio_sys::{
        kAudioDevicePropertyDataSource, kAudioDevicePropertyTransportType,
        kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyScopeOutput, kAudioObjectSystemObject, AudioDeviceID,
        AudioObjectGetPropertyData, AudioObjectHasProperty, AudioObjectID,
        AudioObjectPropertyAddress,
    };

    use super::{verdict, Route};

    /// `kAudioObjectPropertyElementMain` (nee `...Master`) is always `0`;
    /// spelled out as a literal so this keeps compiling across SDKs that rename
    /// it. Same reasoning as `detect/macos.rs`.
    const ELEMENT_MAIN: u32 = 0;
    const NO_DEVICE: AudioDeviceID = 0;

    pub(super) fn address(selector: u32, scope: u32) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress {
            mSelector: selector,
            mScope: scope,
            mElement: ELEMENT_MAIN,
        }
    }

    fn has_property(object_id: AudioObjectID, address: &AudioObjectPropertyAddress) -> bool {
        // SAFETY: `address` is a valid, initialized address that we own;
        // `AudioObjectHasProperty` only reads `object_id`/`address` and never
        // writes through either pointer.
        unsafe { AudioObjectHasProperty(object_id, address) != 0 }
    }

    /// One `UInt32`-valued property, or `None` when it cannot be asked or
    /// answered on this object.
    ///
    /// Keeping "no" and "I could not find out" apart is the whole point, here
    /// as in `detect/macos.rs`: a zero read out of a failed call would be a
    /// transport of `0` — `kAudioDeviceTransportTypeUnknown` — which is a
    /// perfectly ordinary answer and would be judged as one.
    pub(super) fn read_u32(
        object_id: AudioObjectID,
        address: &AudioObjectPropertyAddress,
    ) -> Option<u32> {
        if object_id == NO_DEVICE || !has_property(object_id, address) {
            return None;
        }
        let mut value: u32 = 0;
        let mut size = mem::size_of::<u32>() as u32;
        // SAFETY: `address` is a valid, initialized `AudioObjectPropertyAddress`;
        // `value` is a valid `u32`-sized buffer we own for the duration of the
        // call and `size` matches it exactly. CoreAudio writes at most `size`
        // bytes and reports how much it actually wrote.
        let status = unsafe {
            AudioObjectGetPropertyData(
                object_id,
                address as *const AudioObjectPropertyAddress,
                0,
                std::ptr::null(),
                &mut size,
                &mut value as *mut u32 as *mut c_void,
            )
        };
        if status != 0 || size as usize != mem::size_of::<u32>() {
            return None;
        }
        Some(value)
    }

    pub(super) fn default_output_device() -> Option<AudioDeviceID> {
        let address = address(
            kAudioHardwarePropertyDefaultOutputDevice,
            kAudioObjectPropertyScopeGlobal,
        );
        read_u32(kAudioObjectSystemObject, &address).filter(|id| *id != NO_DEVICE)
    }

    /// The transport of the default output device, and the data source of that
    /// device **in the output scope**.
    ///
    /// The scope is load-bearing and it is the one thing in this file that is
    /// easy to get wrong: `kAudioDevicePropertyDataSource` read in the *input*
    /// scope of a MacBook's built-in device answers about the microphone, not
    /// about the jack, and it will happily hand back a code that has nothing to
    /// do with where the sound is coming out. The transport, by contrast, is a
    /// property of the whole device and is asked globally.
    pub(super) fn read_route() -> Route {
        let Some(device) = default_output_device() else {
            return Route::CannotTell;
        };
        let transport = read_u32(
            device,
            &address(
                kAudioDevicePropertyTransportType,
                kAudioObjectPropertyScopeGlobal,
            ),
        );
        let data_source = read_u32(
            device,
            &address(
                kAudioDevicePropertyDataSource,
                kAudioObjectPropertyScopeOutput,
            ),
        );
        verdict(transport, data_source)
    }
}

/// What this machine is playing through right now.
///
/// Two property reads, nothing held. Anything that cannot be read is
/// [`Route::CannotTell`], which arms — see the module docs.
#[cfg(target_os = "macos")]
pub fn current() -> Route {
    ffi::read_route()
}

/// Nothing to ask on this platform, so nothing is claimed.
///
/// [`Route::CannotTell`] arms, so bleed detection on Linux rides entirely on
/// the predicate — which is where the safety actually lives. Reading the
/// PipeWire node's port type here is the natural follow-up.
#[cfg(not(target_os = "macos"))]
pub fn current() -> Route {
    Route::CannotTell
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The table in [`verdict`], row by row, because the table *is* the
    /// judgement and the rest of this file is two property reads.
    #[test]
    fn the_built_in_output_is_read_off_its_data_source() {
        assert_eq!(
            verdict(Some(TRANSPORT_BUILT_IN), Some(DATA_SOURCE_INTERNAL_SPEAKER)),
            Route::Loudspeaker
        );
        assert_eq!(
            verdict(Some(TRANSPORT_BUILT_IN), Some(DATA_SOURCE_EXTERNAL_SPEAKER)),
            Route::Loudspeaker,
            "powered speakers in the jack are still a loudspeaker in a room"
        );
        assert_eq!(
            verdict(Some(TRANSPORT_BUILT_IN), Some(DATA_SOURCE_HEADPHONES)),
            Route::Headphones
        );
        assert_eq!(
            verdict(Some(TRANSPORT_BUILT_IN), None),
            Route::CannotTell,
            "a built-in output that will not say which way it is pointing is \
             not evidence of headphones"
        );
        assert_eq!(
            verdict(Some(TRANSPORT_BUILT_IN), Some(fourcc(*b"none"))),
            Route::CannotTell
        );
    }

    #[test]
    fn a_screen_or_a_speaker_on_the_network_plays_into_a_room() {
        for transport in [TRANSPORT_HDMI, TRANSPORT_DISPLAY_PORT, TRANSPORT_AIR_PLAY] {
            // The data source is irrelevant on these, including when it is
            // missing — a television has no jack to report.
            assert_eq!(verdict(Some(transport), None), Route::Loudspeaker);
            assert_eq!(
                verdict(Some(transport), Some(DATA_SOURCE_HEADPHONES)),
                Route::Loudspeaker,
                "the transport decides on these; nothing plugs headphones into \
                 an AirPlay speaker"
            );
        }
    }

    /// The row that decides how much this file can be trusted: on these
    /// transports AirPods and a speakerphone on a meeting-room table are the
    /// *same two numbers*, so the honest answer is that nobody knows — and the
    /// honest answer arms.
    #[test]
    fn bluetooth_and_usb_and_virtual_devices_cannot_be_told_apart() {
        for transport in [
            fourcc(*b"blue"),
            fourcc(*b"blea"),
            fourcc(*b"usb "),
            fourcc(*b"virt"),
            fourcc(*b"grup"),
        ] {
            assert_eq!(verdict(Some(transport), None), Route::CannotTell);
            assert_eq!(
                verdict(Some(transport), Some(DATA_SOURCE_HEADPHONES)),
                Route::CannotTell,
                "a Bluetooth driver claiming a headphone data source is still \
                 a driver we have no reason to believe"
            );
        }
    }

    #[test]
    fn an_unreadable_device_is_never_headphones() {
        assert_eq!(verdict(None, None), Route::CannotTell);
        assert_eq!(
            verdict(None, Some(DATA_SOURCE_HEADPHONES)),
            Route::CannotTell,
            "a data source with no transport behind it says nothing: the same \
             four characters mean different things on different drivers"
        );
        assert_eq!(verdict(Some(0), None), Route::CannotTell);
    }

    /// Only headphones disarm, and the reason is physics rather than taste.
    #[test]
    fn only_headphones_disarm() {
        assert!(Route::Loudspeaker.arms());
        assert!(Route::CannotTell.arms());
        assert!(!Route::Headphones.arms());
    }

    /// The four-character codes, pinned. `ispk`/`espk`/`hdpn` are **not** in
    /// any header — they are what the built-in output driver actually reports —
    /// so the only thing standing between this file and a silent mix-up is this
    /// test, the way `the_stream_direction_constant_matches_the_header` is in
    /// `detect/macos.rs`.
    #[test]
    fn the_four_character_codes_are_the_ones_apple_uses() {
        assert_eq!(DATA_SOURCE_INTERNAL_SPEAKER, 0x6973_706B); // 'i','s','p','k'
        assert_eq!(DATA_SOURCE_EXTERNAL_SPEAKER, 0x6573_706B); // 'e','s','p','k'
        assert_eq!(DATA_SOURCE_HEADPHONES, 0x6864_706E); // 'h','d','p','n'
                                                         // Big-endian, not little: get this backwards and every code in the file
                                                         // silently becomes a different code that no driver will ever report.
        assert_eq!(fourcc(*b"bltn").to_be_bytes(), *b"bltn");
        // …and the three are distinct, which is what the match arms rely on.
        assert_ne!(DATA_SOURCE_INTERNAL_SPEAKER, DATA_SOURCE_EXTERNAL_SPEAKER);
        assert_ne!(DATA_SOURCE_INTERNAL_SPEAKER, DATA_SOURCE_HEADPHONES);
    }

    /// The transports are spelled out locally so the judgement is portable;
    /// this is what keeps the local spelling honest against the generated
    /// bindings.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_transports_match_the_generated_bindings() {
        assert_eq!(
            TRANSPORT_BUILT_IN,
            coreaudio_sys::kAudioDeviceTransportTypeBuiltIn
        );
        assert_eq!(TRANSPORT_HDMI, coreaudio_sys::kAudioDeviceTransportTypeHDMI);
        assert_eq!(
            TRANSPORT_DISPLAY_PORT,
            coreaudio_sys::kAudioDeviceTransportTypeDisplayPort
        );
        assert_eq!(
            TRANSPORT_AIR_PLAY,
            coreaudio_sys::kAudioDeviceTransportTypeAirPlay
        );
        // The ones the table deliberately does not name, so that a future
        // reader can see they were considered rather than forgotten.
        assert_eq!(
            fourcc(*b"blue"),
            coreaudio_sys::kAudioDeviceTransportTypeBluetooth
        );
        assert_eq!(
            fourcc(*b"blea"),
            coreaudio_sys::kAudioDeviceTransportTypeBluetoothLE
        );
        assert_eq!(
            fourcc(*b"usb "),
            coreaudio_sys::kAudioDeviceTransportTypeUSB
        );
        assert_eq!(
            fourcc(*b"virt"),
            coreaudio_sys::kAudioDeviceTransportTypeVirtual
        );
        assert_eq!(
            fourcc(*b"grup"),
            coreaudio_sys::kAudioDeviceTransportTypeAggregate
        );
    }

    /// The output scope has to reach CoreAudio as a different address from the
    /// input one, or the comment about the microphone is just a comment.
    #[cfg(target_os = "macos")]
    #[test]
    fn the_data_source_is_asked_about_the_output_half() {
        use coreaudio_sys::{
            kAudioDevicePropertyDataSource, kAudioObjectPropertyScopeInput,
            kAudioObjectPropertyScopeOutput,
        };
        let output = ffi::address(
            kAudioDevicePropertyDataSource,
            kAudioObjectPropertyScopeOutput,
        );
        let input = ffi::address(
            kAudioDevicePropertyDataSource,
            kAudioObjectPropertyScopeInput,
        );
        assert_eq!(output.mSelector, input.mSelector);
        assert_ne!(
            output.mScope, input.mScope,
            "the data source in the input scope is the microphone, not the jack"
        );
        assert_eq!(output.mScope, kAudioObjectPropertyScopeOutput);
    }

    /// Not a test: a way to look at what this machine's CoreAudio actually
    /// says, which is the only way to check the FFI half of this file. Run it
    /// with headphones in and then with them out:
    ///
    /// ```text
    /// cargo test audio::route::tests::what_this_machine_plays_through -- --ignored --nocapture
    /// ```
    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "diagnostic; prints what this machine's CoreAudio reports"]
    fn what_this_machine_plays_through() {
        use coreaudio_sys::{
            kAudioDevicePropertyDataSource, kAudioDevicePropertyTransportType,
            kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyScopeOutput,
        };

        fn spell(code: Option<u32>) -> String {
            match code {
                None => "unreadable".to_string(),
                Some(value) => {
                    let bytes = value.to_be_bytes();
                    let text: String = bytes.iter().map(|b| *b as char).collect();
                    format!("{value} ({text:?})")
                }
            }
        }

        let device = ffi::default_output_device();
        println!("default output device: {device:?}");
        if let Some(device) = device {
            let transport = ffi::read_u32(
                device,
                &ffi::address(
                    kAudioDevicePropertyTransportType,
                    kAudioObjectPropertyScopeGlobal,
                ),
            );
            let data_source = ffi::read_u32(
                device,
                &ffi::address(
                    kAudioDevicePropertyDataSource,
                    kAudioObjectPropertyScopeOutput,
                ),
            );
            println!("  transport:    {}", spell(transport));
            println!("  data source:  {}", spell(data_source));
            println!("  verdict:      {:?}", verdict(transport, data_source));
        }
        println!("current() -> {:?} (arms: {})", current(), current().arms());
    }
}
