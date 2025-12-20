//! # memo-stt
//!
//! **Plug-and-play speech-to-text for Rust applications.**
//!
//! Add voice transcription to your app in 3 lines of code. No API keys, no cloud services,
//! no configuration. Just audio in, text out.
//!
//! ## Quick Example
//!
//! ```no_run
//! use memo_stt::SttEngine;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = SttEngine::new_default(16000)?;
//! engine.warmup()?;
//! let text = engine.transcribe(&audio_samples)?;
//! println!("Transcribed: {}", text);
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - ✅ **Zero Configuration** - Works immediately after `cargo add memo-stt`
//! - ✅ **GPU Accelerated** - Automatic Metal/CUDA acceleration
//! - ✅ **Cross-Platform** - macOS, Windows, Linux
//! - ✅ **Simple API** - Just 3 methods: `new()`, `warmup()`, `transcribe()`
//! - ✅ **Fast** - Sub-second transcription latency
//! - ✅ **Private** - All processing happens locally
//!
//! ## Use Cases
//!
//! - Voice commands in desktop apps
//! - Dictation in text editors
//! - Accessibility tools
//! - Real-time transcription
//! - Voice assistants
//!
//! ## Installation
//!
//! ```toml
//! [dependencies]
//! memo-stt = "0.1"
//! ```
//!
//! ## Model Setup
//!
//! Models are automatically downloaded on first use! No manual setup required.
//! The default model (`ggml-small.en-q5_1.bin`, ~500MB) will be downloaded to your
//! cache directory when you first call `SttEngine::new_default()`.
//!
//! ## Comparison
//!
//! | Feature | memo-stt | whisper-rs | OpenAI API |
//! |---------|----------|------------|------------|
//! | Setup | ✅ Zero config | ❌ Complex | ❌ API keys |
//! | Privacy | ✅ Local | ✅ Local | ❌ Cloud |
//! | Cost | ✅ Free | ✅ Free | ❌ Per request |
//! | Speed | ✅ Fast | ✅ Fast | ⚠️ Network latency |
//! | GPU | ✅ Auto | ✅ Manual | N/A |

pub mod engine;
pub mod model;

pub use engine::SttEngine;
pub use model::{default_model_path, ensure_model};

/// Default Whisper model name (small.en Q5_1)
/// 
/// The model will be automatically downloaded to the cache directory on first use.
pub const DEFAULT_MODEL: &str = "ggml-small.en-q5_1.bin";

/// Simple error type
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
