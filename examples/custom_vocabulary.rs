//! Custom-vocabulary example.
//!
//! The engine accepts an initial prompt that biases transcription toward
//! specific terms, useful for product names, jargon, code identifiers, etc.
//!
//! Run with:
//! ```bash
//! cargo run --example custom_vocabulary
//! ```

use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = SttEngine::new_default(16000)?;

    engine.set_prompt(Some(
        "Rust programming language, cargo, crates.io, GitHub, \
         async await, tokio, serde, clippy, rustfmt"
            .to_string(),
    ));

    engine.warmup()?;

    println!("Engine ready with custom vocabulary.");
    println!("Pass audio samples to engine.transcribe(&samples) to use it.");

    Ok(())
}
