//! GUI integration example pattern
//!
//! This example shows how to integrate memo-stt into GUI applications.
//! The pattern works with any GUI framework (egui, iced, tauri, etc.).
//!
//! This is a conceptual example - adapt to your specific GUI framework.

use memo_stt::SttEngine;

// Example: Button click handler pattern
fn handle_record_button_click() -> Result<String, Box<dyn std::error::Error>> {
    // In a real app, you'd create the engine once and reuse it
    let mut engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;
    engine.warmup()?;
    
    // Get audio samples from your audio capture system
    let samples: Vec<i16> = capture_audio()?;
    
    // Transcribe
    let text = engine.transcribe(&samples)?;
    
    Ok(text)
}

// Example: Real-time transcription pattern
fn setup_realtime_transcription() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;
    engine.warmup()?;
    
    // In your audio stream callback:
    // audio_stream.on_chunk(|samples| {
    //     let text = engine.transcribe(&samples)?;
    //     update_ui_text(text);
    // });
    
    Ok(())
}

// Placeholder for audio capture (implement based on your needs)
fn capture_audio() -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    // Replace with your actual audio capture implementation
    // For example, using cpal, rodio, or your framework's audio API
    Ok(vec![])
}

fn main() {
    println!("GUI Integration Example");
    println!("\nThis example shows integration patterns for GUI applications.");
    println!("Key patterns:");
    println!("  1. Create engine once, reuse for multiple transcriptions");
    println!("  2. Call warmup() after creation for better performance");
    println!("  3. Transcribe in button handlers or audio callbacks");
    println!("  4. Update UI with transcribed text");
    println!("\nSee the source code for implementation details.");
}
