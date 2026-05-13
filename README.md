# memo-stt

Plug-and-play speech-to-text for Rust. Add local transcription to any app in a
few lines, with automatic GPU acceleration and zero configuration. Avoid
expensive API calls.

[![crates.io](https://img.shields.io/crates/v/memo-stt.svg)](https://crates.io/crates/memo-stt)
[![docs.rs](https://docs.rs/memo-stt/badge.svg)](https://docs.rs/memo-stt)
[![downloads](https://img.shields.io/crates/d/memo-stt.svg)](https://crates.io/crates/memo-stt)
[![CI](https://github.com/oliverbhull/memo-stt/actions/workflows/ci.yml/badge.svg)](https://github.com/oliverbhull/memo-stt/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick start

```toml
[dependencies]
memo-stt = "0.1"
```

```rust
use memo_stt::SttEngine;

let mut engine = SttEngine::new_default(16000)?;
engine.warmup()?;
let text = engine.transcribe(&audio_samples)?;
println!("Transcribed: {}", text);
```

On the first call, the default model
(`ggml-small.en-q5_1.bin`, ~500 MB) is downloaded to your platform cache
directory. Every subsequent run is fully offline.

## Why memo-stt

- **Zero configuration.** No API keys, no environment variables, no
  manual model setup.
- **Local and private.** Audio never leaves the machine.
- **Automatic GPU acceleration.** Metal on macOS; CUDA on Linux/Windows when
  available; clean CPU fallback otherwise.
- **Simple, three-method API.** `new_default` / `warmup` / `transcribe`.
- **Cross-platform.** macOS, Linux, Windows.

## Recommended model

> **Use `ggml-small.en-q5_1.bin` (the default).** It is the best general-purpose
> choice for almost every use case: ~500 MB on disk, sub-second latency on
> modern hardware, and accuracy that is very close to the larger distil models
> for clean English speech.

You only need a different model if you have a specific reason:

| Model | Size | Typical latency (M1) | When to use |
|-------|------|---------------------|-------------|
| `ggml-small.en-q5_1` *(default)* | ~500 MB | 200–500 ms | **Recommended.** Best balance of speed, size, accuracy. |
| `ggml-distil-large-v3-q5_1` | ~500 MB | 300–600 ms | Noisy audio, accents, harder transcripts. |
| `ggml-distil-large-v3-q8_0` | ~800 MB | 400–800 ms | Maximum accuracy, willing to pay extra latency and disk. |

Models live in your platform cache directory:

- **macOS**: `~/Library/Caches/memo-stt/models/`
- **Linux**: `~/.cache/memo-stt/models/`
- **Windows**: `%LOCALAPPDATA%\memo-stt\models\`

Pre-built models can be downloaded from the
[model repository on Hugging Face](https://huggingface.co/ggerganov/whisper.cpp).

### Quantization, briefly

- **Q5_1** — 5-bit quantization. Smaller, faster, very close to full accuracy
  for English. This is the recommended default.
- **Q8_0** — 8-bit quantization. Larger and slower, slight accuracy bump.

If you are not sure which to pick, pick **Q5_1**. The small.en-q5_1 model is
the sweet spot for nearly all real-time applications.

## Examples

### Basic transcription

```rust
use memo_stt::SttEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut engine = SttEngine::new_default(16000)?;
    engine.warmup()?; // optional, reduces first-call latency

    let samples: Vec<i16> = vec![/* 16 kHz mono PCM */];
    let text = engine.transcribe(&samples)?;
    println!("{}", text);
    Ok(())
}
```

### Custom model path

```rust
use memo_stt::SttEngine;

let engine = SttEngine::new("models/ggml-small.en-q5_1.bin", 16000)?;
```

### Custom vocabulary / context prompt

```rust
use memo_stt::SttEngine;

let mut engine = SttEngine::new_default(16000)?;
engine.set_prompt(Some("Rust, cargo, crates.io, tokio".to_string()));
engine.warmup()?;
```

More examples live in the [`examples/`](examples/) directory.

## API reference

`SttEngine` — the main transcription engine.

| Method | Purpose |
|--------|---------|
| `SttEngine::new_default(sample_rate)` | Create with the default model (auto-downloaded). |
| `SttEngine::new(model_path, sample_rate)` | Create with a custom model file. |
| `engine.warmup()` | Pre-initialize GPU state to reduce first-call latency. |
| `engine.transcribe(&samples)` | Run inference on 16-bit mono PCM samples. |
| `engine.set_prompt(Some(text))` | Seed transcription with custom vocabulary. |

Full rustdoc is published at [docs.rs/memo-stt](https://docs.rs/memo-stt).

## Audio format

- 16-bit signed PCM (`i16`)
- Mono
- Any sample rate (specified to `new` / `new_default`); resampled to 16 kHz
  internally
- Minimum length: roughly 1 second

## Platform support

| Feature | macOS | Linux | Windows |
|---------|:-----:|:-----:|:-------:|
| Library / `SttEngine` | ✓ | ✓ | ✓ |
| GPU acceleration | Metal | CUDA (if installed) | CUDA (if installed) |
| Standalone binary (mic + hotkeys) | ✓ | ✓ | ✓ |
| Active-application context | ✓ | — | — |

## Requirements

- Rust **1.74** or newer
- ~500 MB of free disk space for the default model
- Internet connection for the one-time model download

## Standalone binary (optional)

memo-stt also ships a CLI with hotkey-driven recording, microphone capture,
and BLE-device support. It is gated behind the `binary` feature so it does
not pull heavy dependencies into library consumers.

```bash
cargo install memo-stt --features binary
```

Then:

```bash
memo-stt                          # default: system mic + Fn hotkey
memo-stt --hotkey Control         # use a different trigger key
INPUT_SOURCE=ble memo-stt         # use a paired BLE audio device
```

### CLI features

- Push-to-talk recording with a configurable hotkey (default: `Fn`)
- Hold-to-lock continuous recording (`Fn` + `Control`)
- Optional BLE audio input from `memo_`-prefixed devices
- Real-time 7-bar waveform output for desktop UI integration
- Active application + window title capture on macOS
- Structured JSON output for downstream tools

### CLI output

The CLI prints a JSON object per transcription:

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

### CLI environment variables

| Variable | Values | Description |
|----------|--------|-------------|
| `INPUT_SOURCE` | `system` (default), `ble`, `radio` | Audio input source. |
| `MEMO_AUDIO_LEVELS_INTERVAL_MS` | `0` (default) or ms | Throttle `AUDIO_LEVELS:` waveform lines. `0` emits every callback. |

### Desktop integration protocol

When embedded in a desktop app, the CLI writes a few well-known stdout lines:

- `AUDIO_LEVELS:<json array>` — 7 waveform values in `0..=1`
- `BLE_PRESS_ENTER` — emitted on BLE control `0x03` (second tap after stop)

## Framework integration

`SttEngine` is `Send` and reusable across calls; create it once and reuse it.

### Tauri

```rust
use memo_stt::SttEngine;

#[tauri::command]
fn transcribe_audio(samples: Vec<i16>) -> Result<String, String> {
    let mut engine = SttEngine::new_default(16000).map_err(|e| e.to_string())?;
    engine.transcribe(&samples).map_err(|e| e.to_string())
}
```

### egui / iced / any GUI framework

```rust
use memo_stt::SttEngine;

// Create the engine once in your app state and reuse it.
let mut engine = SttEngine::new_default(16000)?;
engine.warmup()?;

// In your event/button handler:
let text = engine.transcribe(&audio_samples)?;
```

## Contributing

Issues and pull requests are welcome at
[github.com/oliverbhull/memo-stt](https://github.com/oliverbhull/memo-stt).
Please run `cargo fmt`, `cargo clippy`, and `cargo test` before submitting.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on open-source local speech-recognition runtimes and model tooling.
