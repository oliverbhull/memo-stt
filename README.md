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
- **BLE-connected hardware devices** with voice input
- **Remote trigger devices** for hands-free recording

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

### Platform-Specific Features

| Feature | macOS | Linux | Windows |
|---------|-------|-------|---------|
| GPU Acceleration | ✅ Metal | ⚠️ CUDA (if available) | ⚠️ CUDA (if available) |
| App Context Detection | ✅ Full | ❌ Not yet | ❌ Not yet |
| BLE Support | ✅ | ✅ | ✅ |
| Hotkey Triggers | ✅ | ✅ | ✅ |

**macOS-Only Features**:
- Application context detection captures the active app name and window title
- On Linux/Windows, `appContext` will return "Unknown"

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

### Understanding Model Quantization

The model names include quantization levels (e.g., `q5_1`, `q8_0`):
- **Q5_1**: 5-bit quantization - Smaller size, faster inference, slightly lower accuracy
- **Q8_0**: 8-bit quantization - Larger size, slower inference, highest accuracy

For most use cases, Q5_1 provides the best balance of speed and accuracy.

## 📱 BLE Device Support

memo-stt includes built-in support for Bluetooth Low Energy (BLE) audio devices, enabling wireless voice transcription from hardware buttons or wearable devices.

### Features

- **Wireless Audio Streaming**: Receive Opus-encoded audio directly from BLE devices
- **Remote Trigger Control**: Use hardware buttons to start/stop recording
- **Hybrid Mode**: BLE device as trigger, system microphone for audio
- **Auto-Discovery**: Automatically finds and connects to memo devices

### BLE Configuration

#### Device Discovery

memo-stt automatically discovers BLE devices matching:
- **Device Name Pattern**: `memo_` (prefix)
- **Fallback Address**: `64D5A7E1-B149-191F-9B11-96F5CCF590BF`
- **Scan Timeout**: 30 seconds

#### GATT Service Specification

**Memo Audio Service UUID**: `1234A000-1234-5678-1234-56789ABCDEF0`

**Characteristics**:
1. **Audio Data** (`1234A001-1234-5678-1234-56789ABCDEF0`)
   - Receives Opus-encoded audio at 16kHz mono
   - Frame duration: 20ms
   - Bitrate: 24 kbps (VOIP-optimized)

2. **Control TX** (`1234A003-1234-5678-1234-56789ABCDEF0`)
   - Button events: `0x01` (start), `0x02` (stop)
   - Enables hardware trigger functionality

### BLE Usage Examples

#### Full Audio Mode (Library)

```rust
use memo_stt::ble::BleAudioReceiver;

// Connect to BLE device and receive audio
let receiver = BleAudioReceiver::connect().await?;

// Audio samples are streamed at 16kHz
let samples = receiver.receive_audio().await?;
```

#### Trigger-Only Mode (Library)

```rust
use memo_stt::ble::BleAudioReceiver;

// Use BLE device as trigger, audio from system mic
let receiver = BleAudioReceiver::connect_trigger_only().await?;

// Listen for button press events (0x01, 0x02)
let button_event = receiver.receive_control_event().await?;
```

#### Standalone Binary with BLE

```bash
# Use BLE audio input
INPUT_SOURCE=ble cargo run --bin memo-stt

# Use BLE trigger with system microphone (hybrid mode)
INPUT_SOURCE=ble_trigger cargo run --bin memo-stt
```

## 🖥️ Standalone Binary Application

memo-stt includes a full-featured binary application for hands-free voice transcription with hotkey and BLE device support.

### Installation

```bash
cargo install memo-stt
```

### Running

```bash
# Default mode (system microphone, hotkey trigger)
memo-stt

# With custom hotkey
memo-stt --hotkey Control

# BLE audio mode
INPUT_SOURCE=ble memo-stt

# BLE trigger mode (BLE button, system mic)
INPUT_SOURCE=ble_trigger memo-stt
```

### Features

- **Real-Time Audio Visualization**: 7-bar waveform display
- **Hotkey Recording**: Configurable function key (default: Fn)
- **Lock Mode**: Hold Fn+Control to lock recording on
- **Application Context Detection**: Captures active app and window title (macOS)
- **JSON Output**: Structured transcription with metadata
- **Press-Enter-After-Paste**: Automatically submits text after pasting

### Keyboard Controls

| Action | Keys | Description |
|--------|------|-------------|
| Record | Fn (hold) | Press and hold to record audio |
| Lock Recording | Fn+Control | Toggle continuous recording mode |
| Stop | Release Fn | Stop recording and transcribe |
| Configure Hotkey | `--hotkey <key>` | Change trigger key (e.g., Control, Command) |

### Output Format

The binary outputs JSON with transcription results and context:

```json
{
  "rawTranscript": "Hello world",
  "processedText": "Hello world",
  "wasProcessedByLLM": false,
  "appContext": {
    "appName": "Terminal",
    "windowTitle": "~/dev/memo-stt"
  }
}
```

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `INPUT_SOURCE` | `mic` (default), `ble`, `ble_trigger` | Audio input source |
| `PRESS_ENTER_AFTER_PASTE` | `true`, `false` | Auto-submit after paste |

### Binary Performance

The standalone binary includes additional latency for audio capture and processing:

| Source | Processing Time | Total Latency |
|--------|----------------|---------------|
| System Mic (48kHz) | 50-100ms | 250-600ms |
| BLE Audio (16kHz) | 30-50ms | 230-550ms |
| BLE Trigger + Mic | 50-100ms | 250-600ms |

*Tested on M1 MacBook Pro with default model*

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
