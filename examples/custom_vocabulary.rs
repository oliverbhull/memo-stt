//! Custom vocabulary example
//!
//! This example shows how to use custom prompts to improve accuracy
//! for domain-specific terms, names, or technical vocabulary.
//!
//! Run with: `cargo run --example custom_vocabulary`

use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Creating STT engine with custom vocabulary...");
    
    let mut engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;
    
    // Set custom vocabulary/context to improve accuracy
    // This helps with domain-specific terms, names, technical jargon, etc.
    engine.set_prompt(Some(
        "Rust programming language, cargo, crates.io, GitHub, \
         async await, tokio, serde, clippy, rustfmt".to_string()
    ));
    
    engine.warmup()?;
    
    println!("Engine ready with custom vocabulary!");
    println!("\nThe engine will now be better at recognizing:");
    println!("  - Rust-related terms");
    println!("  - Programming terminology");
    println!("  - Technical vocabulary");
    
    // Example usage:
    // let samples: Vec<i16> = /* your audio */;
    // let text = engine.transcribe(&samples)?;
    // println!("Transcribed: {}", text);
    
    Ok(())
}
