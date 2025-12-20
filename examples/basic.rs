//! Basic transcription example - the simplest possible usage
//!
//! This example shows the absolute minimum code needed to transcribe audio.
//! Run with: `cargo run --example basic`

use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating STT engine...");
    
    // Create engine with default model
    // Note: You'll need to provide a valid model path
    let mut engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;
    
    println!("Warming up GPU...");
    engine.warmup()?;
    
    println!("Ready! Engine is initialized.");
    println!("\nTo transcribe audio, provide PCM samples:");
    println!("  let samples: Vec<i16> = /* your audio data */;");
    println!("  let text = engine.transcribe(&samples)?;");
    println!("  println!(\"Transcribed: {{}}\", text);");
    
    // Example with dummy data (replace with actual audio)
    // let samples: Vec<i16> = vec![0; 16000]; // 1 second of silence
    // let text = engine.transcribe(&samples)?;
    // println!("Transcribed: {}", text);
    
    Ok(())
}
