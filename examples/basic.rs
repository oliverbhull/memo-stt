//! Minimal transcription example.
//!
//! Run with:
//! ```bash
//! cargo run --example basic
//! ```
//!
//! On the first run, the default model (~500 MB) will be downloaded
//! automatically into your platform cache directory.

use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating STT engine (this may download the model on first run)...");
    let mut engine = SttEngine::new_default(16000)?;

    println!("Warming up...");
    engine.warmup()?;

    println!("Engine ready.");
    println!();
    println!("To transcribe, pass 16-bit mono PCM samples:");
    println!("    let samples: Vec<i16> = /* your audio */;");
    println!("    let text = engine.transcribe(&samples)?;");

    // Example with one second of silence (just to demonstrate the call shape).
    let samples = vec![0i16; 16_000];
    match engine.transcribe(&samples) {
        Ok(text) => println!("Transcribed (silence): {:?}", text),
        Err(e) => println!("Transcribe error: {}", e),
    }

    Ok(())
}
