# memo-stt

> **Plug-and-play speech-to-text in 3 lines of code.** Add voice transcription to any Rust application with zero configuration. GPU-accelerated, cross-platform, production-ready.

[![crates.io](https://img.shields.io/crates/v/memo-stt.svg)](https://crates.io/crates/memo-stt)
[![docs.rs](https://docs.rs/memo-stt/badge.svg)](https://docs.rs/memo-stt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Quick Start (3 Lines)

```rust
use memo_stt::SttEngine;

let mut engine = SttEngine::new_default(16000)?;
engine.warmup()?;
let text = engine.transcribe(&audio_samples)?;
// Done! You have transcribed text.
```

## ✨ Why memo-stt?

- **Zero Configuration** - Works out of the box, no API keys, no setup
- **Automatic Model Download** - Models download automatically on first use
- **GPU Accelerated** - Metal (macOS), CUDA (Linux/Windows) support
- **Cross-Platform** - macOS, Windows, Linux
- **Production Ready** - Used in production applications
- **Simple API** - Audio in, text out. That's it.
- **Fast** - 200-500ms transcription latency
- **Private** - Everything runs locally, no cloud calls

## 📦 Installation

```toml
[dependencies]
memo-stt = "0.1"
```

```bash
cargo add memo-stt
```

## 🎯 Use Cases

Perfect for:
- **Voice commands** in desktop applications
- **Dictation** features in text editors
- **Accessibility** tools for voice input
- **Note-taking** apps with voice transcription
- **Voice assistants** and chatbots
- **Real-time transcription** in video/audio apps
- **Accessibility** features requiring voice input

## 📚 Examples

### Basic Transcription

```rust
use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine (model downloads automatically on first use)
    let mut engine = SttEngine::new_default(16000)?;
    
    // Warm up GPU (optional, but recommended)
    engine.warmup()?;
    
    // Your audio samples (16kHz, mono, i16 PCM)
    let samples: Vec<i16> = vec![]; // Replace with actual audio
    let text = engine.transcribe(&samples)?;
    println!("Transcribed: {}", text);
    
    Ok(())
}
```

**That's it!** The model downloads automatically the first time you run this.

### Custom Model Path

```rust
use memo_stt::SttEngine;

// Use a custom model path (won't auto-download unless it's the default model name)
let engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;

// Or use the default (auto-downloads if needed)
let engine = SttEngine::new_default(16000)?;
```

### With Custom Vocabulary

```rust
use memo_stt::SttEngine;

let mut engine = SttEngine::new_default(16000)?;
engine.set_prompt(Some("Rust programming language, cargo, crates.io".to_string()));
engine.warmup()?;
let text = engine.transcribe(&samples)?;
```

### Full Example with Audio Recording

See the [examples directory](examples/) for complete examples including:
- Microphone recording
- Real-time transcription
- GUI integration patterns

## ⚡ Performance

Tested on M1 MacBook Pro:

| Model | Size | Latency | Accuracy |
|-------|------|---------|----------|
| small.en-q5_1 | 500MB | 200-500ms | ⭐⭐⭐⭐ |
| distil-large-v3-q5_1 | 500MB | 300-600ms | ⭐⭐⭐⭐⭐ |
| distil-large-v3-q8_0 | 800MB | 400-800ms | ⭐⭐⭐⭐⭐ |

*Latency measured from audio input to text output*

## 🤔 Why memo-stt vs alternatives?

| Solution | Setup Time | Privacy | Cost | GPU | Ease of Use |
|----------|-----------|---------|------|-----|-------------|
| **memo-stt** | ⚡ 30 seconds | ✅ 100% Local | ✅ Free | ✅ Auto | ⭐⭐⭐⭐⭐ |
| whisper-rs | ⏱️ 2+ hours | ✅ Local | ✅ Free | ⚠️ Manual | ⭐⭐ |
| OpenAI API | ⚡ 5 minutes | ❌ Cloud | 💰 $0.006/min | N/A | ⭐⭐⭐ |
| Google STT | ⚡ 5 minutes | ❌ Cloud | 💰 $0.006/min | N/A | ⭐⭐⭐ |
| AssemblyAI | ⚡ 5 minutes | ❌ Cloud | 💰 $0.00025/sec | N/A | ⭐⭐⭐ |

**memo-stt is the only solution that's:**
- ✅ Zero configuration (models download automatically)
- ✅ 100% private (local processing)
- ✅ Free forever
- ✅ GPU-accelerated automatically
- ✅ Works offline (after initial download)

## 📋 Requirements

- **Rust**: 1.92+
- **Internet**: Required for first-time model download (~500MB)
- **macOS**: Metal GPU acceleration (automatic)
- **Linux/Windows**: CUDA support (if available)

**Note**: After the initial download, models are cached locally and no internet connection is needed.

## 🔧 Model Setup

**Models are automatically downloaded on first use!** No manual setup required.

When you call `SttEngine::new_default()`, the default model (`ggml-small.en-q5_1.bin`, ~500MB) will be automatically downloaded to your cache directory if it doesn't already exist.

### Default Model Location

Models are stored in:
- **macOS**: `~/Library/Caches/memo-stt/models/`
- **Linux**: `~/.cache/memo-stt/models/`
- **Windows**: `%LOCALAPPDATA%\memo-stt\models\`

### Using Custom Models

If you want to use a different model, you can:

1. **Download manually** and provide the path:
   ```rust
   let engine = SttEngine::new("path/to/your/model.bin", 16000)?;
   ```

2. **Recommended models** (download from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp)):
   - `ggml-small.en-q5_1.bin` (~500MB) - Best balance ⭐ **Default**
   - `ggml-distil-large-v3-q5_1.bin` (~500MB) - Higher accuracy
   - `ggml-distil-large-v3-q8_0.bin` (~800MB) - Highest accuracy

## 📖 API Reference

### `SttEngine`

The main transcription engine.

#### Methods

- **`new(model_path, sample_rate)`** - Create engine with custom model
- **`new_default(sample_rate)`** - Create engine with default model path
- **`warmup()`** - Pre-initialize GPU (recommended)
- **`transcribe(samples)`** - Transcribe audio samples to text
- **`set_prompt(prompt)`** - Set custom vocabulary/context

See [full documentation](https://docs.rs/memo-stt) for details.

## 🔌 Framework Integrations

memo-stt works with any Rust framework. Here are some integration patterns:

### Tauri

```rust
use memo_stt::SttEngine;

#[tauri::command]
fn transcribe_audio(samples: Vec<i16>) -> Result<String, String> {
    let mut engine = SttEngine::new_default(16000)
        .map_err(|e| e.to_string())?;
    engine.transcribe(&samples)
        .map_err(|e| e.to_string())
}
```

### egui / Iced / Other GUI Frameworks

```rust
use memo_stt::SttEngine;

// In your button click handler
button.on_click(|| {
    let mut engine = SttEngine::new_default(16000)?;
    let text = engine.transcribe(&audio_samples)?;
    text_field.set_text(text);
});
```

## 🛠️ Audio Format

memo-stt expects audio in the following format:

- **Format**: 16-bit signed integer PCM (`i16`)
- **Channels**: Mono
- **Sample Rate**: Any (specify when creating engine, e.g., 16000, 48000)
- **Minimum Length**: 1 second of audio

The engine automatically handles resampling if your input sample rate differs from 16kHz.

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built on top of:
- [whisper-rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for Whisper
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) - Whisper model inference
- [OpenAI Whisper](https://github.com/openai/whisper) - The original Whisper model

---

**Made with ❤️ by the Memo team**

Questions? Open an issue or check the [documentation](https://docs.rs/memo-stt).
