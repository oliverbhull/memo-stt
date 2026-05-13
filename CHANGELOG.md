# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-12

### Added
- Initial public release.
- `SttEngine` with `new_default`, `new`, `warmup`, `set_prompt`, and
  `transcribe` methods.
- Automatic download of the default `ggml-small.en-q5_1.bin` Whisper model
  into the platform cache directory on first use.
- Optional `binary` feature providing a standalone CLI (`memo-stt`) with
  hotkey-driven microphone capture, BLE audio device support, and structured
  JSON output for desktop integrations.
- Metal GPU acceleration on macOS via `whisper-rs`, with automatic CPU
  fallback on other platforms.

[Unreleased]: https://github.com/oliverbhull/memo-stt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/oliverbhull/memo-stt/releases/tag/v0.1.0
