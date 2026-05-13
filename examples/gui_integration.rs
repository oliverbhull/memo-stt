//! GUI-integration patterns.
//!
//! This file is a sketch — adapt the patterns to your specific GUI
//! framework (egui, iced, tauri, slint, etc.). The key ideas:
//!
//! 1. Create the engine once and reuse it across calls.
//! 2. Call `warmup()` once after construction to reduce first-call latency.
//! 3. Run `transcribe()` off the UI thread (it blocks for a few hundred ms).

use memo_stt::SttEngine;

fn handle_record_button_click(
    engine: &mut SttEngine,
) -> Result<String, Box<dyn std::error::Error>> {
    let samples = capture_audio()?;
    let text = engine.transcribe(&samples)?;
    Ok(text)
}

fn setup_realtime_transcription() -> Result<SttEngine, Box<dyn std::error::Error>> {
    let engine = SttEngine::new_default(16000)?;
    engine.warmup()?;
    Ok(engine)
}

fn capture_audio() -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    // Replace with your audio-capture implementation (cpal, rodio, your
    // framework's audio API, etc.). Must produce 16-bit mono PCM at the
    // sample rate you passed to `SttEngine::new_default`.
    Ok(vec![0i16; 16_000])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = setup_realtime_transcription()?;
    let _ = handle_record_button_click(&mut engine);

    println!("GUI integration sketch — see source for patterns:");
    println!("  1. Create SttEngine once, reuse for many transcriptions.");
    println!("  2. Call warmup() after construction.");
    println!("  3. Call transcribe() off the UI thread.");
    Ok(())
}
