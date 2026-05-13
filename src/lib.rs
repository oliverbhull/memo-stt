//! # memo-stt
//!
//! Plug-and-play speech-to-text for Rust.
//!
//! Add local transcription to any app in a few lines, with automatic GPU
//! acceleration and zero configuration. Avoid expensive API calls.
//!
//! ## Quick example
//!
//! ```no_run
//! use memo_stt::SttEngine;
//!
//! # fn run(audio_samples: &[i16]) -> Result<(), Box<dyn std::error::Error>> {
//! let mut engine = SttEngine::new_default(16000)?;
//! engine.warmup()?;
//! let text = engine.transcribe(audio_samples)?;
//! println!("Transcribed: {}", text);
//! # Ok(())
//! # }
//! ```
//!
//! On the first call, the default model (`ggml-small.en-q5_1.bin`, ~500 MB) is
//! downloaded into the platform cache directory. Every subsequent run is
//! fully offline.
//!
//! ## Recommended model
//!
//! Use the default `ggml-small.en-q5_1.bin` for almost every use case. It is
//! the best general-purpose balance of speed, size, and accuracy for English
//! speech. Only pick a larger distil model if you specifically need higher
//! accuracy on noisy audio or accented speech.
//!
//! ## Features
//!
//! - Zero configuration — model is auto-downloaded on first use.
//! - Automatic GPU acceleration where supported (Metal on macOS, CUDA on
//!   Linux/Windows when available); clean CPU fallback otherwise.
//! - Three-method API: [`SttEngine::new_default`], [`SttEngine::warmup`],
//!   [`SttEngine::transcribe`].
//! - Fully local — audio never leaves the machine.
//! - Cross-platform: macOS, Linux, Windows.
//!
//! ## Installation
//!
//! ```toml
//! [dependencies]
//! memo-stt = "0.1"
//! ```
//!
//! ## Audio format
//!
//! `memo-stt` expects 16-bit signed PCM mono samples (`&[i16]`). The input
//! sample rate is whatever you declare to [`SttEngine::new`] /
//! [`SttEngine::new_default`]; samples are resampled to 16 kHz internally.
//!
//! ## Standalone binary
//!
//! A CLI with hotkey, microphone, and BLE-device support is available behind
//! the `binary` feature:
//!
//! ```bash
//! cargo install memo-stt --features binary
//! ```

pub mod engine;
pub mod model;

pub use engine::SttEngine;
pub use model::{default_model_path, ensure_model};

/// Default model name (`small.en` Q5_1).
///
/// This is the recommended general-purpose model and is downloaded
/// automatically on first use.
pub const DEFAULT_MODEL: &str = "ggml-small.en-q5_1.bin";

/// Error type used throughout the crate.
#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for Error {}

/// Convenience `Result` alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
