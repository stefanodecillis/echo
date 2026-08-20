//! Print the tensor contract of an ONNX file — input and output names, declared
//! shapes and element types — and optionally run it on a fixed signal so two
//! exports can be compared for functional identity.
//!
//! Echo's decoders read the contract at load time rather than assuming it
//! ([`echo_lib::diarize::segmentation::declared_rank`],
//! [`echo_lib::diarize::segmentation::powerset_classes`]), so this is how a
//! candidate model asset is checked against what those decoders can handle
//! *before* it goes in the catalog.
//!
//! ```text
//! cargo run --release --example onnx_contract -- [--run] <file.onnx> […]
//!
//!   --run   also push 10 s of a deterministic pseudo-random 16 kHz signal
//!           through each model and print a checksum of the output. Two exports
//!           that print the same checksum are the same network.
//! ```

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use echo_lib::diarize::features;
use echo_lib::diarize::segmentation::{declared_rank, WINDOW_SAMPLES};

/// A deterministic "audio-like" signal: no file to keep, same bytes every run.
fn signal(n: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|i| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let noise = ((state >> 40) as f32 / 8_388_608.0) - 1.0;
            let tone = (i as f32 * 0.02).sin();
            0.2 * tone + 0.05 * noise
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut run = false;
    let mut paths: Vec<String> = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--run" {
            run = true;
        } else {
            paths.push(arg);
        }
    }

    for path in paths {
        println!("=== {path}");
        let mut session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .commit_from_file(&path)?;

        let mut input_name = String::new();
        let mut rank = 3usize;
        let mut takes_fbank = false;
        for input in session.inputs() {
            println!("  in   {:<20} {:?}", input.name(), input.dtype());
            if input_name.is_empty() {
                input_name = input.name().to_string();
                rank = declared_rank(input.dtype()).unwrap_or(3);
                takes_fbank =
                    format!("{:?}", input.dtype()).contains(&format!("{}", features::NUM_MEL_BINS));
            }
        }
        for output in session.outputs() {
            println!("  out  {:<20} {:?}", output.name(), output.dtype());
        }

        if run {
            let audio = signal(WINDOW_SAMPLES);
            let (shape, data) = if takes_fbank {
                let feats = features::fbank(&audio);
                let frames = feats.frames as i64;
                let bins = features::NUM_MEL_BINS as i64;
                let shape = if rank <= 2 {
                    vec![frames, bins]
                } else {
                    vec![1, frames, bins]
                };
                (shape, feats.data)
            } else {
                let samples = audio.len() as i64;
                let shape = match rank {
                    0 | 1 => vec![samples],
                    2 => vec![1, samples],
                    _ => vec![1, 1, samples],
                };
                (shape, audio)
            };
            let tensor = Tensor::from_array((shape, data))?;
            let outputs = session.run(ort::inputs![input_name.as_str() => tensor])?;
            let (out_shape, values) = outputs[0].try_extract_tensor::<f32>()?;
            let sum: f64 = values.iter().map(|v| f64::from(*v)).sum();
            let abs: f64 = values.iter().map(|v| f64::from(v.abs())).sum();
            println!(
                "  run  shape {:?}  n {}  sum {sum:.6}  abs-sum {abs:.6}",
                out_shape.iter().copied().collect::<Vec<i64>>(),
                values.len()
            );
        }
        println!();
    }
    Ok(())
}
