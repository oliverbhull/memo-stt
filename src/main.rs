//! memo-stt binary - Standalone speech-to-text application
//!
//! This binary provides a complete STT application with keyboard triggers.
//! For library usage, see the examples directory.

use memo_stt::SttEngine;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rdev::{listen, Event, EventType, Key};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc;
use std::time::Instant;
use std::collections::{HashMap, VecDeque};
use serde_json::json;
#[cfg(feature = "binary")]
use log::debug;
mod app_detection;

/// When stdout is a pipe (Electron), Rust uses a block buffer — lines can sit until the buffer fills.
/// Flush so the UI overlay sees recording / stopped state immediately.
macro_rules! println_ui_flush {
    ($($arg:tt)*) => {{
        println!($($arg)*);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }};
}

/// Minimum interval between `AUDIO_LEVELS:` lines (ms). `0` = no throttle (emit every callback / decoded frame).
/// Set `MEMO_AUDIO_LEVELS_INTERVAL_MS` (e.g. `33` for ~30 fps) if stdout/IPC cannot keep up.
static MEMO_AUDIO_LEVELS_INTERVAL_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

fn memo_audio_levels_interval_ms() -> u64 {
    *MEMO_AUDIO_LEVELS_INTERVAL_MS.get_or_init(|| {
        std::env::var("MEMO_AUDIO_LEVELS_INTERVAL_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    })
}

fn should_emit_audio_levels_throttled(last_sent: &mut Option<Instant>, interval_ms: u64) -> bool {
    if interval_ms == 0 {
        return true;
    }
    let now = Instant::now();
    match last_sent {
        None => {
            *last_sent = Some(now);
            true
        }
        Some(prev) if now.duration_since(*prev).as_millis() >= u128::from(interval_ms) => {
            *last_sent = Some(now);
            true
        }
        Some(_) => false,
    }
}

#[cfg(feature = "binary")]
mod ble;
#[cfg(feature = "binary")]
mod opus_decoder;

/// Trailing phrases often triggered by button/PTT click sounds — strip from end of transcript.
const SIGN_OFF_PHRASES: &[&str] = &[
    "thank you",
    "thanks",
    "thanks for watching",
    "bye",
    "goodbye",
];

/// Strip trailing sign-off phrases (e.g. "Thank you.", "Bye", "Thanks for watching") from transcript.
/// These are often falsely triggered by the sound of a button/PTT click at end of recording.
fn strip_trailing_signoffs(text: &str) -> String {
    let mut out = text.trim().to_string();
    if out.is_empty() {
        return out;
    }
    loop {
        let prev_len = out.len();
        let out_trimmed = out.trim_end_matches(|c: char| c == '.' || c == ',' || c == ' ' || c == '!');
        let out_lower = out_trimmed.to_lowercase();
        for phrase in SIGN_OFF_PHRASES {
            if out_lower.ends_with(phrase) {
                let n = out_trimmed.chars().count();
                let p_len = phrase.chars().count();
                if n >= p_len {
                    let cut = n - p_len;
                    out = out_trimmed.chars().take(cut).collect::<String>();
                    out = out.trim_end_matches(|c: char| c == ' ' || c == '.' || c == ',').to_string();
                    break;
                }
            }
        }
        if out.len() == prev_len {
            break;
        }
    }
    out.trim_end_matches(|c: char| c == ' ' || c == ',').to_string()
}

/// Strip trailing period from short final phrase (<4 words).
/// Internal sentence-ending punctuation is always preserved to maintain readability.
fn strip_periods_from_short_phrases(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let last_char = trimmed.chars().last().unwrap();
    if last_char != '.' && last_char != '!' && last_char != '?' {
        return trimmed.to_string();
    }

    let without_final = &trimmed[..trimmed.len() - last_char.len_utf8()];
    let last_delim = without_final.rfind(|c: char| c == '.' || c == '!' || c == '?');
    let last_sentence = match last_delim {
        Some(pos) => &without_final[pos + 1..],
        None => without_final,
    };
    let word_count = last_sentence.trim().split_whitespace().count();
    if word_count < 4 {
        return without_final.to_string();
    }

    trimmed.to_string()
}

/// Strip leading dash and following space(s) from transcript (e.g. Whisper bullet-style "- Can you...").
fn strip_leading_dash_space(text: &str) -> String {
    let s = text.trim();
    if s.starts_with('-') {
        s[1..].trim_start().to_string()
    } else {
        s.to_string()
    }
}

/// Join streaming transcription segments with proper sentence boundaries.
/// Ensures each non-final segment ends with punctuation so sentences don't run together.
fn join_segments(parts: &[String]) -> String {
    if parts.is_empty() { return String::new(); }
    if parts.len() == 1 { return parts[0].clone(); }

    let mut result = String::new();
    for (i, part) in parts.iter().enumerate() {
        let trimmed = part.trim();
        if trimmed.is_empty() { continue; }
        if !result.is_empty() { result.push(' '); }
        result.push_str(trimmed);
        if i < parts.len() - 1 {
            let last = trimmed.chars().last().unwrap_or(' ');
            if last != '.' && last != '!' && last != '?' && last != ',' {
                result.push('.');
            }
        }
    }
    result
}

// Calculate audio levels for waveform visualization
// Returns 7 normalized levels (0.0-1.0) for the 7 bars
fn calculate_audio_levels(samples: &[i16]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; 7];
    }
    
    // Calculate RMS (Root Mean Square) for audio level
    let sum_squares: i64 = samples.iter().map(|&s| (s as i64).pow(2)).sum();
    let rms = (sum_squares as f32 / samples.len() as f32).sqrt();
    
    // Normalize to 0-1 range (i16 max is 32767)
    // Use lower threshold and gain boost for better reactivity (similar to memo-desktop system mic)
    const NORMALIZATION_THRESHOLD: f32 = 15000.0;
    const GAIN_BOOST: f32 = 2.0;
    let normalized = ((rms / NORMALIZATION_THRESHOLD) * GAIN_BOOST).min(1.0);
    
    // Apply exponential scaling for better visual response
    let scaled = normalized.powf(0.4);
    
    // Create 7 bands with symmetric weighting (center bars higher, edges taper down)
    let weights = vec![0.6, 0.8, 0.95, 1.0, 0.95, 0.8, 0.6];
    weights.into_iter()
        .map(|w| (scaled * w).min(1.0))
        .collect()
}

// Calculate audio levels for BLE waveform overlay (0.0–1.0 per bar).
// Calibrated for firmware 20ms frames / current PDM gain; tune via env if needed.
fn calculate_audio_levels_ble(samples: &[i16]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; 7];
    }

    let sum_squares: i64 = samples.iter().map(|&s| (s as i64).pow(2)).sum();
    let rms = (sum_squares as f32 / samples.len() as f32).sqrt();

    // Normalize to 0–1 for overlay. Threshold/gain tuned for BLE decoded PCM (20ms bundles).
    // Optional env override: MEMO_BLE_WAVEFORM_THRESHOLD, MEMO_BLE_WAVEFORM_GAIN
    let threshold = std::env::var("MEMO_BLE_WAVEFORM_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(20000.0);
    let gain = std::env::var("MEMO_BLE_WAVEFORM_GAIN")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.5);
    let normalized = ((rms / threshold) * gain).min(1.0);

    let scaled = normalized.powf(0.4);
    let weights = vec![0.6, 0.8, 0.95, 1.0, 0.95, 0.8, 0.6];
    weights.into_iter().map(|w| (scaled * w).min(1.0)).collect()
}
#[cfg(not(target_os = "macos"))]
use enigo::{Enigo, KeyboardControllable, Key as EnigoKey};

// Default trigger key (can be overridden via --hotkey argument)
const DEFAULT_TRIGGER_KEY: Key = Key::Function;

/// Resolve input device: "default", numeric index, or substring name match (e.g. "AirPods", "External Microphone").
fn find_input_device_by_spec(host: &cpal::Host, spec: &str) -> Option<cpal::Device> {
    let spec = spec.trim();
    if spec.is_empty() || spec.eq_ignore_ascii_case("default") {
        return host.default_input_device();
    }
    let devices: Vec<cpal::Device> = match host.input_devices() {
        Ok(iter) => iter.collect(),
        Err(_) => return host.default_input_device(),
    };
    if let Ok(idx) = spec.parse::<usize>() {
        if idx < devices.len() {
            return Some(devices[idx].clone());
        }
        return host.default_input_device();
    }
    let spec_lower = spec.to_lowercase();
    for dev in &devices {
        if let Ok(name) = dev.name() {
            if name.to_lowercase().contains(&spec_lower) {
                return Some(dev.clone());
            }
        }
    }
    host.default_input_device()
}

/// Prefer mono input and the highest sample rate up to 48 kHz (better quality when the device allows it).
fn best_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig, Box<dyn std::error::Error>> {
    use cpal::SampleRate;
    let default = device.default_input_config()?;
    let Ok(mut configs) = device.supported_input_configs() else {
        return Ok(default);
    };
    let candidates: Vec<cpal::SupportedStreamConfigRange> = configs.collect();
    if candidates.is_empty() {
        return Ok(default);
    }
    let pick_mono = candidates
        .iter()
        .filter(|c| c.channels() == 1)
        .max_by_key(|c| c.max_sample_rate().0);
    let pick_any = candidates
        .iter()
        .max_by_key(|c| c.max_sample_rate().0);
    let range = pick_mono.or(pick_any).unwrap_or(&candidates[0]);
    let max_sr = range.max_sample_rate().0;
    let min_sr = range.min_sample_rate().0;
    let target = max_sr.min(48_000).max(min_sr);
    Ok(range.with_sample_rate(SampleRate(target)))
}

fn extend_buffer_mono_i16(buf: &mut Vec<i16>, data: &[i16], channels: usize) {
    match channels {
        1 => buf.extend_from_slice(data),
        n if n > 1 => {
            for frame in data.chunks_exact(n) {
                let sum: i32 = frame.iter().map(|&s| s as i32).sum();
                buf.push((sum / n as i32) as i16);
            }
        }
        _ => {}
    }
}

fn extend_buffer_mono_f32(buf: &mut Vec<i16>, data: &[f32], channels: usize) {
    match channels {
        1 => {
            for &s in data {
                buf.push((s.clamp(-1.0, 1.0) * 32767.0) as i16);
            }
        }
        n if n > 1 => {
            for frame in data.chunks_exact(n) {
                let mut acc = 0.0f32;
                for &s in frame {
                    acc += s.clamp(-1.0, 1.0);
                }
                buf.push(((acc / n as f32) * 32767.0) as i16);
            }
        }
        _ => {}
    }
}

fn extend_buffer_mono_u16(buf: &mut Vec<i16>, data: &[u16], channels: usize) {
    match channels {
        1 => {
            for &s in data {
                buf.push(((s as i32) - 32768) as i16);
            }
        }
        n if n > 1 => {
            for frame in data.chunks_exact(n) {
                let sum: i32 = frame.iter().map(|&s| (s as i32) - 32768).sum();
                buf.push((sum / n as i32) as i16);
            }
        }
        _ => {}
    }
}

fn audio_levels_interleaved_i16(data: &[i16], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        return calculate_audio_levels(data);
    }
    let mono: Vec<i16> = data
        .chunks_exact(ch)
        .map(|frame| {
            let s: i32 = frame.iter().map(|&x| x as i32).sum();
            (s / ch as i32) as i16
        })
        .collect();
    calculate_audio_levels(&mono)
}

fn audio_levels_interleaved_f32(data: &[f32], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        let mono: Vec<i16> = data
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
            .collect();
        return calculate_audio_levels(&mono);
    }
    let mono: Vec<i16> = data
        .chunks_exact(ch)
        .map(|frame| {
            let acc: f32 = frame.iter().map(|&s| s.clamp(-1.0, 1.0)).sum::<f32>() / ch as f32;
            (acc * 32767.0) as i16
        })
        .collect();
    calculate_audio_levels(&mono)
}

fn audio_levels_interleaved_u16(data: &[u16], ch: usize) -> Vec<f32> {
    if ch <= 1 {
        let mono: Vec<i16> = data
            .iter()
            .map(|&s| ((s as i32) - 32768) as i16)
            .collect();
        return calculate_audio_levels(&mono);
    }
    let mono: Vec<i16> = data
        .chunks_exact(ch)
        .map(|frame| {
            let s: i32 = frame.iter().map(|&x| (x as i32) - 32768).sum();
            (s / ch as i32) as i16
        })
        .collect();
    calculate_audio_levels(&mono)
}

// Parse hotkey from string to Key enum
fn parse_hotkey(key_str: &str) -> Option<Key> {
    match key_str.to_lowercase().as_str() {
        "function" | "fn" => Some(Key::Function),
        "f1" => Some(Key::F1),
        "f2" => Some(Key::F2),
        "f3" => Some(Key::F3),
        "f4" => Some(Key::F4),
        "f5" => Some(Key::F5),
        "f6" => Some(Key::F6),
        "f7" => Some(Key::F7),
        "f8" => Some(Key::F8),
        "f9" => Some(Key::F9),
        "f10" => Some(Key::F10),
        "f11" => Some(Key::F11),
        "f12" => Some(Key::F12),
        "space" => Some(Key::Space),
        "controlleft" | "ctrl" => Some(Key::ControlLeft),
        "controlright" => Some(Key::ControlRight),
        "altleft" | "altright" | "alt" => Some(Key::Alt),
        "metaleft" | "cmd" | "command" => Some(Key::MetaLeft),
        "metaright" => Some(Key::MetaRight),
        "shiftleft" | "shift" => Some(Key::ShiftLeft),
        "shiftright" => Some(Key::ShiftRight),
        _ => None,
    }
}

// Message types for the channel
#[derive(Debug, Clone, Copy)]
enum KeyEvent {
    StartRecording,
    StopRecording,
    ToggleLock,
}

/// Compute RMS (root mean square) of i16 samples for VAD.
fn compute_rms(samples: &[i16]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: i64 = samples.iter().map(|&s| (s as i64).pow(2)).sum();
    (sum_squares as f32 / samples.len() as f32).sqrt()
}

// Calculate the rate of increase in realtime factor per second of audio
fn calculate_rate_of_increase(history: &[(f32, f32)]) -> Option<f32> {
    if history.len() < 2 {
        return None;
    }
    
    // Simple linear regression: calculate slope (rate of increase)
    let n = history.len() as f32;
    let sum_x: f32 = history.iter().map(|(x, _)| x).sum();
    let sum_y: f32 = history.iter().map(|(_, y)| y).sum();
    let sum_xy: f32 = history.iter().map(|(x, y)| x * y).sum();
    let sum_x2: f32 = history.iter().map(|(x, _)| x * x).sum();
    
    let denominator = n * sum_x2 - sum_x * sum_x;
    if denominator.abs() < 1e-6 {
        return None;
    }
    
    let slope = (n * sum_xy - sum_x * sum_y) / denominator;
    Some(slope)
}

fn inject_text(text: &str, press_enter: bool) -> Result<(), Box<dyn std::error::Error>> {
    if text.trim().is_empty() {
        return Ok(());
    }

    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        if let Some(stdin) = child.stdin.as_mut() {
            stdin.write_all(text.as_bytes())?;
        }
        child.wait()?;
        // Use status() instead of output() - we don't need the output, just execution
        let script = r#"tell application "System Events"
  keystroke "v" using command down
end tell"#;
        std::process::Command::new("osascript")
            .arg("-e")
            .arg(script)
            .status()?;
        
        // Press Enter after paste if enabled
        if press_enter {
            let enter_script = r#"tell application "System Events"
  key code 36
end tell"#;
            std::process::Command::new("osascript")
                .arg("-e")
                .arg(enter_script)
                .status()?;
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        let mut enigo = Enigo::new();
        let paste_mod = EnigoKey::Control;
        enigo.key_down(paste_mod);
        enigo.key_click(EnigoKey::Layout('v'));
        enigo.key_up(paste_mod);
        
        // Press Enter after paste if enabled
        if press_enter {
            enigo.key_click(EnigoKey::Return);
        }
    }
    
    Ok(())
}

#[cfg(feature = "binary")]
async fn run_ble_audio_mode(engine: Arc<Mutex<SttEngine>>, no_inject: bool) -> Result<(), Box<dyn std::error::Error>> {
    use ble::BleAudioReceiver;
    use opus_decoder::OpusDecoder;

    let no_inject_flag = Arc::new(AtomicBool::new(no_inject));

    println!("Starting BLE audio mode...");
    
    // Initialize Opus decoder (preserved during reconnection)
    let mut decoder = OpusDecoder::new(16000, 20)?;
    
    // Initialize BLE receiver
    let mut ble_receiver = BleAudioReceiver::new().await?;
    
    // DO NOT auto-connect - wait for CONNECT_UID command from Electron
    // This prevents duplicate connections
    println!("BLE mode started. Waiting for CONNECT_UID command...");
    
    // State that persists across reconnections (preserved during reconnection)
    let engine_clone = engine.clone();
    let performance_history: Arc<Mutex<VecDeque<(f32, f32)>>> = Arc::new(Mutex::new(VecDeque::with_capacity(10)));
    let press_enter_after_paste = Arc::new(AtomicBool::new(false));
    let is_recording = Arc::new(AtomicBool::new(false));
    let audio_buffer = Arc::new(Mutex::new(Vec::<i16>::new()));
    
    // Vocabulary storage for voice commands
    #[derive(Clone)]
    struct Vocabulary {
        app_names: Vec<String>,
        app_commands: HashMap<String, Vec<String>>,
        global_commands: Vec<String>,
    }
    
    let vocabulary = Arc::new(Mutex::new(Vocabulary {
        app_names: Vec::new(),
        app_commands: HashMap::new(),
        global_commands: Vec::new(),
    }));
    
    // Build prompt with only the active app's commands (+ global) to avoid hallucination
    let build_prompt = |app_name: String, window_title: String, vocab: &Vocabulary| -> Option<String> {
        let mut parts = Vec::new();
        
        if !app_name.is_empty() && app_name != "Unknown" {
            if !window_title.is_empty() {
                parts.push(format!("You are transcribing for {}. The current window is: {}.", app_name, window_title));
            } else {
                parts.push(format!("You are transcribing for {}.", app_name));
            }
        }
        
        if !vocab.app_names.is_empty() {
            parts.push(format!("Voice commands: open {}.", vocab.app_names.join(", ")));
        }
        
        let mut active_cmds: Vec<&str> = vocab.global_commands.iter().map(|s| s.as_str()).collect();
        if !app_name.is_empty() {
            let key = app_name.to_lowercase();
            if let Some(cmds) = vocab.app_commands.get(&key) {
                active_cmds.extend(cmds.iter().map(|s| s.as_str()));
            }
        }
        if !active_cmds.is_empty() {
            parts.push(format!("Commands: {}.", active_cmds.join(", ")));
        }
        
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };
    
    // Channel for connection commands from stdin handler
    let (connect_tx, mut connect_rx) = tokio::sync::mpsc::unbounded_channel::<Option<String>>();
    
    // Spawn thread to read commands from stdin
    let press_enter_clone = press_enter_after_paste.clone();
    let input_source = Arc::new(Mutex::new(String::from("ble")));
    let input_source_clone = input_source.clone();
    let vocabulary_clone = vocabulary.clone();
    let connect_tx_for_stdin = connect_tx.clone();
    std::thread::spawn(move || {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(cmd) = line {
                if let Some(value) = cmd.strip_prefix("ENTER:") {
                    let enable = value.trim() == "1" || value.trim().eq_ignore_ascii_case("true");
                    press_enter_clone.store(enable, Ordering::Release);
                    eprintln!("MIC: Press-Enter-after-paste: {}", enable);
                } else if let Some(value) = cmd.strip_prefix("INPUT_SOURCE:") {
                    let source = value.trim().to_lowercase();
                    if source == "system" || source == "ble" {
                        let source_clone = source.clone();
                        *input_source_clone.lock().unwrap() = source;
                        eprintln!("MIC: Input source changed to: {}", source_clone);
                    }
                } else if let Some(uid) = cmd.strip_prefix("CONNECT_UID:") {
                    let uid = uid.trim().to_uppercase();
                    eprintln!("MIC: Connecting to device with UID: {}", uid);
                    // Format device name: memo_XXXXX
                    let device_name = format!("memo_{}", uid);
                    // Send connection request via channel
                    let _ = connect_tx_for_stdin.send(Some(device_name));
                } else if cmd.trim() == "DISCONNECT" {
                    eprintln!("MIC: Disconnecting from BLE device");
                    // Send disconnect request via channel (None means disconnect)
                    let _ = connect_tx_for_stdin.send(None);
                } else if let Some(value) = cmd.strip_prefix("VOCAB:") {
                    if let Ok(vocab_json) = serde_json::from_str::<serde_json::Value>(value.trim()) {
                        let mut vocab = vocabulary_clone.lock().unwrap();
                        vocab.app_names = vocab_json.get("appNames")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.global_commands = vocab_json.get("globalCommands")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.app_commands.clear();
                        if let Some(obj) = vocab_json.get("appCommands").and_then(|v| v.as_object()) {
                            for (key, val) in obj {
                                if let Some(arr) = val.as_array() {
                                    let cmds: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                                    vocab.app_commands.insert(key.clone(), cmds);
                                }
                            }
                        }
                        eprintln!("MIC: Vocabulary updated: {} app names, {} apps with commands, {} global commands",
                            vocab.app_names.len(), vocab.app_commands.len(), vocab.global_commands.len());
                    } else {
                        eprintln!("MIC: Failed to parse VOCAB command");
                    }
                }
            }
        }
    });
    
    // Main loop: listen for button presses and buffer audio during recording
    let is_recording_clone = is_recording.clone();
    let audio_buffer_clone = audio_buffer.clone();
    
    use ble::NotificationResult;
    use futures::StreamExt;
    use tokio::time::timeout;
    
    // Outer loop: wait for CONNECT_UID command, then connect and process notifications
    loop {
        // Check if input source changed
        {
            let source = input_source.lock().unwrap();
            if *source != "ble" {
                println!("Input source changed to {}, exiting BLE mode", source);
                break;
            }
        }
        
        // Wait for CONNECT_UID command - DO NOT auto-connect
        let device_name = loop {
            match connect_rx.recv().await {
                Some(Some(name)) => {
                    // Got connect command
                    break Some(name);
                }
                Some(None) => {
                    // Got disconnect command
                    eprintln!("Disconnecting from BLE device");
                    ble_receiver.disconnect().await.ok();
                    println!("DISCONNECTED:user_requested");
                    return Ok(());
                }
                None => {
                    // Channel closed
                    return Ok(());
                }
            }
        };
        
        let device_name = match device_name {
            Some(name) => name,
            None => continue,
        };
        
        // Connect to device
        eprintln!("Connecting to device: {}", device_name);
        if let Err(e) = ble_receiver.connect(Some(&device_name)).await {
            eprintln!("Failed to connect to device {}: {}", device_name, e);
            println!("DISCONNECTED:connection_failed");
            continue; // Wait for next CONNECT_UID command
        }
        
        // Get the notification stream
        let mut notifications = match ble_receiver.notifications().await {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!("Failed to get notification stream: {}", e);
                println!("DISCONNECTED:notification_stream_failed");
                // Break to outer loop - wait for new CONNECT_UID command
                break;
            }
        };
        
        // Inner loop: process notifications with connection monitoring
        // Track last notification time to detect real disconnections
        let mut last_notification_time = std::time::Instant::now();
        // Central-side light polling (low power): confirm link health while idle via a cheap GATT read.
        // This avoids false disconnects when the device is connected but idle (no notifications).
        let poll_interval_secs: u64 = std::env::var("MEMO_BLE_POLL_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(15);
        let poll_failures_before_disconnect: u32 = std::env::var("MEMO_BLE_POLL_FAILURES_BEFORE_DISCONNECT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        let mut poll_interval = tokio::time::interval(tokio::time::Duration::from_secs(poll_interval_secs));
        poll_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut poll_failure_count: u32 = 0;

        // Faster detection during active recording: if we expect notifications but they stop, run a health check.
        let mut recording_health_check_interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        recording_health_check_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut recording_health_failure_count: u32 = 0;

        // Track expected bundle index for packet-loss detection; use FEC when previous packet was lost.
        let mut expected_bundle_index: Option<u8> = None;
        let last_audio_level_sent_ble = Arc::new(Mutex::new(None::<Instant>));

        loop {
            // Check if input source changed
            {
                let source = input_source.lock().unwrap();
                if *source != "ble" {
                    println!("Input source changed to {}, exiting BLE mode", source);
                    return Ok(());
                }
            }
            
            // Use select! to monitor notifications, connection health, and connection commands
            tokio::select! {
                // Low-frequency poll while idle: a small GATT read to confirm the link is alive.
                _ = poll_interval.tick() => {
                    if ble_receiver.poll_link().await {
                        poll_failure_count = 0;
                    } else {
                        poll_failure_count = poll_failure_count.saturating_add(1);
                        eprintln!(
                            "BLE poll failed ({}/{}) - may be disconnected",
                            poll_failure_count, poll_failures_before_disconnect
                        );
                        if poll_failure_count >= poll_failures_before_disconnect {
                            eprintln!("BLE poll failures exceeded threshold, assuming connection_lost");
                            println!("DISCONNECTED:connection_lost");
                            break;
                        }
                    }
                }
                // Check for connection/disconnect commands (non-blocking)
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {
                    match connect_rx.try_recv() {
                        Ok(Some(device_name)) => {
                            // Check if already connected to the same device
                            if ble_receiver.is_connected() {
                                if let Some(current_name) = ble_receiver.device_name() {
                                    if current_name == &device_name {
                                        // Already connected to same device, skip reconnect
                                        continue;
                                    }
                                }
                            }
                            // Need to reconnect - break inner loop
                            eprintln!("Reconnecting to device: {}", device_name);
                            ble_receiver.disconnect().await.ok();
                            if let Err(e) = ble_receiver.connect(Some(&device_name)).await {
                                eprintln!("Failed to connect: {}", e);
                                println!("DISCONNECTED:connection_failed");
                            }
                            break; // Break inner loop to get new notification stream
                        }
                        Ok(None) => {
                            // Disconnect
                            ble_receiver.disconnect().await.ok();
                            println!("DISCONNECTED:user_requested");
                            return Ok(());
                        }
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                            // No command available, continue
                        }
                        Err(_) => {
                            // Channel closed, exit
                            return Ok(());
                        }
                    }
                }
                // Fast detection during active recording: if notifications stop for a while, confirm link health.
                _ = recording_health_check_interval.tick() => {
                    if is_recording_clone.load(Ordering::Acquire) {
                        let time_since_last_notification = last_notification_time.elapsed();
                        if time_since_last_notification.as_secs() >= 8 {
                            if !ble_receiver.check_connection_health().await {
                                recording_health_failure_count = recording_health_failure_count.saturating_add(1);
                                eprintln!(
                                    "Recording health check failed ({}/2), no notifications for {}s",
                                    recording_health_failure_count,
                                    time_since_last_notification.as_secs()
                                );
                                if recording_health_failure_count >= 2 {
                                    eprintln!("Recording health check failures exceeded threshold, assuming connection_lost");
                                    println!("DISCONNECTED:connection_lost");
                                    break;
                                }
                            } else {
                                recording_health_failure_count = 0;
                            }
                        } else {
                            recording_health_failure_count = 0;
                        }
                    } else {
                        recording_health_failure_count = 0;
                    }
                }
                
                // Receive notifications (audio or control events) with timeout
                result = timeout(tokio::time::Duration::from_millis(100), notifications.next()) => {
                    match result {
                        Ok(Some(notification)) => {
                            // Update last notification time - we're actively receiving data
                            last_notification_time = std::time::Instant::now();
                            poll_failure_count = 0;
                            recording_health_failure_count = 0;
                            
                            // Process the notification
                            match ble_receiver.process_notification(notification) {
                                NotificationResult::Control(0x01) => {
                                    // RESP_SPEECH_START - Button pressed, start recording
                                    if !is_recording_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                        continue; // Already recording
                                    }
                                    println_ui_flush!("🎤 Recording... (button pressed)");
                                    audio_buffer_clone.lock().unwrap().clear();
                                }
                                NotificationResult::Control(0x02) => {
                                    // RESP_SPEECH_END - Button pressed again, stop recording and transcribe
                                    if !is_recording_clone.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                        continue; // Not recording
                                    }
                                    
                                    // Get the buffered audio
                                    let samples = {
                                        let mut buf = audio_buffer_clone.lock().unwrap();
                                        std::mem::take(&mut *buf)
                                    };
                                    
                                    if !samples.is_empty() {
                                        println_ui_flush!("⏹️  Stopped ({} samples, {:.2}s)", samples.len(), samples.len() as f32 / 16000.0);
                                        
                                        // Encode audio to OPUS for saving
                                        let samples_for_encoding = samples.clone();
                                        let audio_duration = samples.len() as f32 / 16000.0;
                                        
                                        // Encode in a separate thread to avoid blocking transcription
                                        std::thread::spawn(move || {
                            use opus_decoder::OpusEncoder;
                            #[cfg(feature = "binary")]
                            use base64::{Engine as _, engine::general_purpose::STANDARD};
                            
                            match OpusEncoder::new(16000, 20) {
                                Ok(mut encoder_for_thread) => {
                                    match encoder_for_thread.encode_buffer(&samples_for_encoding) {
                                        Ok(opus_data) => {
                                            #[cfg(feature = "binary")]
                                            {
                                                let base64_data = STANDARD.encode(&opus_data);
                                                println!("AUDIO_DATA:{}", base64_data);
                                                println!("AUDIO_DURATION:{:.2}", audio_duration);
                                                
                                                // Also output WAV data for easy playback
                                                // WAV format: 44-byte header + PCM data
                                                let sample_rate = 16000u32;
                                                let channels = 1u16;
                                                let bits_per_sample = 16u16;
                                                let pcm_data_len = samples_for_encoding.len() * 2; // 16-bit = 2 bytes per sample
                                                let wav_size = 44 + pcm_data_len;
                                                
                                                let mut wav_data = Vec::with_capacity(wav_size);
                                                // RIFF header
                                                wav_data.extend_from_slice(b"RIFF");
                                                wav_data.extend_from_slice(&(36u32 + pcm_data_len as u32).to_le_bytes());
                                                wav_data.extend_from_slice(b"WAVE");
                                                // fmt chunk
                                                wav_data.extend_from_slice(b"fmt ");
                                                wav_data.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
                                                wav_data.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
                                                wav_data.extend_from_slice(&channels.to_le_bytes());
                                                wav_data.extend_from_slice(&sample_rate.to_le_bytes());
                                                wav_data.extend_from_slice(&(sample_rate as u32 * channels as u32 * (bits_per_sample as u32 / 8)).to_le_bytes()); // byte rate
                                                wav_data.extend_from_slice(&(channels * (bits_per_sample / 8)).to_le_bytes()); // block align
                                                wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());
                                                // data chunk
                                                wav_data.extend_from_slice(b"data");
                                                wav_data.extend_from_slice(&(pcm_data_len as u32).to_le_bytes());
                                                // PCM data (16-bit little-endian)
                                                for &sample in &samples_for_encoding {
                                                    wav_data.extend_from_slice(&sample.to_le_bytes());
                                                }
                                                
                                                let wav_base64 = STANDARD.encode(&wav_data);
                                                println!("AUDIO_WAV:{}", wav_base64);
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to encode audio: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("Failed to create Opus encoder: {}", e);
                                }
                            }
                        });
                        
                        // Spawn transcription thread
                        let engine_for_thread = engine_clone.clone();
                        let perf_history = performance_history.clone();
                        let press_enter_clone = press_enter_after_paste.clone();
                        let no_inject_clone = no_inject_flag.clone();
                        let vocabulary_for_thread = vocabulary.clone();
                        let sample_count = samples.len();
                        let audio_duration = sample_count as f32 / 16000.0;
                        
                        std::thread::spawn(move || {
                            println!("🔄 Transcribing...");
                            let mut eng = engine_for_thread.lock().unwrap();
                            
                            // Capture application context and vocabulary
                            let (app_name, window_title) = app_detection::get_application_context();
                            let vocab = vocabulary_for_thread.lock().unwrap();
                            let prompt = build_prompt(app_name, window_title, &vocab);
                            eng.set_prompt(prompt);
                            
                            let transcribe_start = Instant::now();
                            match eng.transcribe(&samples) {
                                Ok(text) => {
                                    let transcribe_time = transcribe_start.elapsed();
                                    let realtime_factor = audio_duration / transcribe_time.as_secs_f32();
                                    
                                    // Update performance history
                                    {
                                        let mut history = perf_history.lock().unwrap();
                                        history.push_back((audio_duration, realtime_factor));
                                        if history.len() > 10 {
                                            history.pop_front();
                                        }
                                    }
                                    
                                    if text.trim().is_empty() {
                                        println!("📝 (no speech detected)");
                                    } else {
                                        let (app_name, window_title) = app_detection::get_application_context();
                                        let processed_text = strip_leading_dash_space(&strip_trailing_signoffs(&strip_periods_from_short_phrases(&text)));
                                        let json_output = json!({
                                            "rawTranscript": text,
                                            "processedText": processed_text,
                                            "wasProcessedByLLM": false,
                                            "appContext": {
                                                "appName": app_name,
                                                "windowTitle": window_title
                                            }
                                        });
                                        println!("FINAL: {}", json_output);
                                        
                                        // Only inject if not in Electron mode
                                        if !no_inject_clone.load(Ordering::Acquire) {
                                            let press_enter = press_enter_clone.load(Ordering::Acquire);
                                            match inject_text(&processed_text, press_enter) {
                                                Ok(_) => {
                                                    println!("📝 {}", text);
                                                    println!("✅ Injected");
                                                }
                                                Err(e) => {
                                                    println!("📝 {}", text);
                                                    eprintln!("❌ Injection failed: {}", e);
                                                }
                                            }
                                        } else {
                                            println!("📝 {}", text);
                                            println!("⏭️  Injection skipped (Electron mode)");
                                        }
                                    }
                                }
                                Err(e) => {
                                    eprintln!("❌ Error: {}", e);
                                }
                            }
                        });
                                    } else {
                                        println_ui_flush!("⏹️  Stopped (no audio captured)");
                                    }
                                }
                                NotificationResult::Control(0x03) => {
                                    // RESP_PRESS_ENTER — second button tap shortly after stop (Memo Desktop)
                                    println_ui_flush!("BLE_PRESS_ENTER");
                                }
                                NotificationResult::Audio(audio_data) => {
                                    // Only process audio if we're recording
                                    if !is_recording_clone.load(Ordering::Acquire) {
                                        continue;
                                    }
                                    
                                    if audio_data.is_empty() {
                                        continue;
                                    }
                                    
                                    // Parse packet: [bundle_index:1][num_frames:1][frame1_size:1][frame1_data:N]...
                                    // Format from firmware: bundle_index (1 byte) + bundled data
                                    if audio_data.len() < 2 {
                                        continue;
                                    }
                                    
                                    // Extract bundle data (skip 1-byte bundle_index header)
                                    let bundle_index = audio_data[0];
                                    let bundle_data = &audio_data[1..];

                                    debug!("Received audio packet: bundle_index={}, size={} bytes", bundle_index, audio_data.len());

                                    let decode_result = match expected_bundle_index {
                                        None => {
                                            expected_bundle_index = Some(bundle_index.wrapping_add(1));
                                            decoder.decode_bundle(bundle_data)
                                        }
                                        Some(expected) => {
                                            if bundle_index == expected {
                                                expected_bundle_index = Some(bundle_index.wrapping_add(1));
                                                decoder.decode_bundle(bundle_data)
                                            } else {
                                                // Gap: one or more packets lost
                                                let lost = bundle_index.wrapping_sub(expected);
                                                for _ in 0..lost.saturating_sub(1) {
                                                    if let Ok(plc) = decoder.decode_plc() {
                                                        audio_buffer_clone.lock().unwrap().extend_from_slice(&plc);
                                                    }
                                                }
                                                expected_bundle_index = Some(bundle_index.wrapping_add(1));
                                                decoder.decode_bundle_with_fec(bundle_data)
                                            }
                                        }
                                    };

                                    match decode_result {
                                        Ok(pcm_samples) => {
                                            if !pcm_samples.is_empty() {
                                                // Add to buffer while recording
                                                audio_buffer_clone.lock().unwrap().extend_from_slice(&pcm_samples);

                                                // Calculate and send audio levels for waveform visualization (BLE with reduced sensitivity)
                                                if is_recording_clone.load(Ordering::Acquire) {
                                                    let levels = calculate_audio_levels_ble(&pcm_samples);
                                                    let interval_ms = memo_audio_levels_interval_ms();
                                                    let mut last_sent = last_audio_level_sent_ble.lock().unwrap();
                                                    if should_emit_audio_levels_throttled(
                                                        &mut *last_sent,
                                                        interval_ms,
                                                    ) {
                                                        let json = json!(levels).to_string();
                                                        println_ui_flush!("AUDIO_LEVELS:{}", json);
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Opus decode error: {}", e);
                                        }
                                    }
                                }
                                NotificationResult::Control(code) => {
                                    eprintln!("BLE unknown control notification: 0x{:02X}", code);
                                }
                                NotificationResult::None => {
                                    // Not a notification we care about
                                }
                            }
                        }
                        Ok(None) => {
                            // Stream ended - attempt reconnection
                            eprintln!("BLE notification stream ended");
                            break; // Break inner loop to attempt reconnection
                        }
                        Err(_) => {
                            // Timeout - no notification available, continue loop
                        }
                    }
                }
            }
        }
        
        // Stream ended or failed - disconnect and exit
        eprintln!("BLE device disconnected");
        println!("DISCONNECTED:connection_lost");
        ble_receiver.disconnect().await.ok();
        break;
    }
    
    // Clean up on exit
    ble_receiver.disconnect().await.ok();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check INPUT_SOURCE environment variable
    let input_source = std::env::var("INPUT_SOURCE").unwrap_or_else(|_| "system".to_string());
    
    // Parse command line arguments for hotkey and no-inject flag
    let args: Vec<String> = std::env::args().collect();
    let mut trigger_key = DEFAULT_TRIGGER_KEY;
    let mut no_inject = false;
    
    for i in 0..args.len() {
        if args[i] == "--hotkey" && i + 1 < args.len() {
            if let Some(key) = parse_hotkey(&args[i + 1]) {
                trigger_key = key;
                println!("Using hotkey: {:?}", trigger_key);
            } else {
                eprintln!("Warning: Unknown hotkey '{}', using default (Function)", args[i + 1]);
            }
        } else if args[i] == "--no-inject" {
            no_inject = true;
            println!("Auto-injection disabled (Electron mode)");
        }
    }
    
    // Branch based on input source
    if input_source == "ble" {
        #[cfg(feature = "binary")]
        {
            // BLE audio is 16kHz, so initialize engine with 16kHz
            println!("Loading Whisper model (16kHz for BLE audio)...");
            let engine = SttEngine::new_default(16000)?;
            
            println!("Warming up GPU...");
            engine.warmup()?;
            println!("Ready!");
            
            let engine_arc = Arc::new(Mutex::new(engine));
            let rt = tokio::runtime::Runtime::new()?;
            return rt.block_on(run_ble_audio_mode(engine_arc, no_inject));
        }
        #[cfg(not(feature = "binary"))]
        {
            eprintln!("BLE mode requires binary feature");
            return Err("BLE mode not available".into());
        }
    }
    
    // Resolve input device and stream config BEFORE creating the STT engine so input_sample_rate matches
    // the actual hardware (critical for Bluetooth / AirPods HFP at 8–16 kHz vs built-in at 48 kHz).
    let host = cpal::default_host();
    let device = if input_source == "radio" {
        let radio_spec = std::env::var("MEMO_RADIO_INPUT_DEVICE")
            .unwrap_or_else(|_| "External Microphone".to_string());
        find_input_device_by_spec(&host, &radio_spec)
            .ok_or_else(|| format!("No input device found for Radio (spec: {:?}). Set MEMO_RADIO_INPUT_DEVICE=0 or device name.", radio_spec))?
    } else {
        let spec = std::env::var("MEMO_SYSTEM_INPUT_DEVICE").unwrap_or_default();
        if !spec.trim().is_empty() {
            find_input_device_by_spec(&host, spec.trim())
                .or_else(|| host.default_input_device())
                .ok_or("No input device found")?
        } else {
            host.default_input_device().ok_or("No input device found")?
        }
    };

    let config = best_input_config(&device)?;
    let sample_rate = config.sample_rate().0;
    let stream_channels = config.channels() as usize;
    let dev_name = device.name().unwrap_or_else(|_| "?".to_string());
    println!("MIC_INFO:{}\t{}", dev_name.replace('\t', " "), sample_rate);
    println!("Using: {}", dev_name);
    println!("Sample rate: {} Hz, channels: {}, format: {:?}", sample_rate, stream_channels, config.sample_format());

    println!("Loading Whisper model ({} Hz input)...", sample_rate);
    let engine = SttEngine::new_default(sample_rate)?;

    println!("Warming up GPU...");
    engine.warmup()?;
    println!("Ready!");

    let engine = Arc::new(Mutex::new(engine));
    let audio_buffer = Arc::new(Mutex::new(Vec::<i16>::new()));
    let is_recording = Arc::new(AtomicBool::new(false));
    let is_locked = Arc::new(AtomicBool::new(false));
    let recording_stream: Arc<Mutex<Option<cpal::Stream>>> = Arc::new(Mutex::new(None));
    let performance_history: Arc<Mutex<VecDeque<(f32, f32)>>> = Arc::new(Mutex::new(VecDeque::with_capacity(10)));
    let press_enter_after_paste = Arc::new(AtomicBool::new(false));
    let no_inject_flag = Arc::new(AtomicBool::new(no_inject));

    let streaming_enabled = std::env::var("STREAMING_TRANSCRIBE")
        .map(|v| v != "0" && v.to_lowercase() != "false")
        .unwrap_or(true);
    let seg_silence_threshold: f32 = std::env::var("SEGMENT_SILENCE_THRESHOLD")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(600.0);
    let seg_silence_ms: u64 = std::env::var("SEGMENT_SILENCE_MS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(1200);
    let seg_min_duration_ms: u64 = std::env::var("SEGMENT_MIN_DURATION_MS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(15000);
    let seg_max_duration_ms: u64 = 28000;
    let segment_results: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let segment_boundary: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let segmenter_active = Arc::new(AtomicBool::new(false));
    let last_segment_text: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let audio_buffer_clone = audio_buffer.clone();
    let is_recording_clone = is_recording.clone();
    let is_locked_clone = is_locked.clone();
    let recording_stream_clone = recording_stream.clone();
    let device_clone = device.clone();
    let config_clone = config.clone();
    let stream_ch = stream_channels;
    let engine_clone = engine.clone();
    let segment_results_clone = segment_results.clone();
    let segment_boundary_clone = segment_boundary.clone();
    let segmenter_active_clone = segmenter_active.clone();
    let last_segment_text_clone = last_segment_text.clone();
    let performance_history_clone = performance_history.clone();
    
    let (tx, rx) = mpsc::channel::<KeyEvent>();

    let use_vad_trigger = input_source == "radio";

    if use_vad_trigger {
        let vad_ch = stream_ch;
        // Radio mode: VAD trigger. One continuous stream feeds both VAD and recording buffer.
        let vad_buffer: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::with_capacity(48000)));
        const VAD_BUFFER_MAX_SAMPLES: usize = 48000; // 1 second at 48 kHz
        let vad_speech_threshold: f32 = std::env::var("VAD_SPEECH_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(800.0);
        let vad_silence_threshold: f32 = std::env::var("VAD_SILENCE_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(600.0);
        let vad_speech_start_ms: u64 = std::env::var("VAD_SPEECH_START_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);
        let vad_silence_ms: u64 = std::env::var("VAD_SILENCE_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1200);
        let vad_poll_interval_ms: u64 = 50;
        let samples_per_poll = (sample_rate as u64 * vad_poll_interval_ms / 1000) as usize;

        let vad_buffer_for_stream = vad_buffer.clone();
        let vad_buffer_for_poll = vad_buffer.clone();
        let audio_buffer_vad = audio_buffer_clone.clone();
        let is_recording_vad = is_recording_clone.clone();
        let device_vad = device_clone.clone();
        let config_vad = config_clone.clone();
        let tx_vad = tx.clone();
        let last_audio_level_sent_vad = Arc::new(Mutex::new(None::<Instant>));

        // Thread 1: continuous stream — push to vad_buffer and to audio_buffer when recording; send AUDIO_LEVELS for waveform
        std::thread::spawn(move || {
            let stream_config = config_vad.clone().into();
            let last_audio_level_sent_clone = last_audio_level_sent_vad.clone();
            let stream_result = match config_vad.sample_format() {
                cpal::SampleFormat::I16 => device_vad.build_input_stream(
                    &stream_config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        let mut mono_frame = Vec::with_capacity(data.len() / vad_ch.max(1));
                        extend_buffer_mono_i16(&mut mono_frame, data, vad_ch);
                        {
                            let mut buf = vad_buffer_for_stream.lock().unwrap();
                            buf.extend(mono_frame.iter().copied());
                            while buf.len() > VAD_BUFFER_MAX_SAMPLES {
                                buf.pop_front();
                            }
                        }
                        if is_recording_vad.load(Ordering::Acquire) {
                            audio_buffer_vad
                                .lock()
                                .unwrap()
                                .extend_from_slice(&mono_frame);
                            let levels = calculate_audio_levels(&mono_frame);
                            let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                            if should_emit_audio_levels_throttled(
                                &mut *last_sent,
                                memo_audio_levels_interval_ms(),
                            ) {
                                let json = json!(levels).to_string();
                                println_ui_flush!("AUDIO_LEVELS:{}", json);
                            }
                        }
                    },
                    |err| eprintln!("VAD stream error: {}", err),
                    None,
                ),
                cpal::SampleFormat::F32 => device_vad.build_input_stream(
                    &stream_config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let mut mono_frame = Vec::with_capacity(data.len() / vad_ch.max(1));
                        extend_buffer_mono_f32(&mut mono_frame, data, vad_ch);
                        {
                            let mut buf = vad_buffer_for_stream.lock().unwrap();
                            buf.extend(mono_frame.iter().copied());
                            while buf.len() > VAD_BUFFER_MAX_SAMPLES {
                                buf.pop_front();
                            }
                        }
                        if is_recording_vad.load(Ordering::Acquire) {
                            audio_buffer_vad
                                .lock()
                                .unwrap()
                                .extend_from_slice(&mono_frame);
                            let levels = calculate_audio_levels(&mono_frame);
                            let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                            if should_emit_audio_levels_throttled(
                                &mut *last_sent,
                                memo_audio_levels_interval_ms(),
                            ) {
                                let json = json!(levels).to_string();
                                println_ui_flush!("AUDIO_LEVELS:{}", json);
                            }
                        }
                    },
                    |err| eprintln!("VAD stream error: {}", err),
                    None,
                ),
                cpal::SampleFormat::U16 => device_vad.build_input_stream(
                    &stream_config,
                    move |data: &[u16], _: &cpal::InputCallbackInfo| {
                        let mut mono_frame = Vec::with_capacity(data.len() / vad_ch.max(1));
                        extend_buffer_mono_u16(&mut mono_frame, data, vad_ch);
                        {
                            let mut buf = vad_buffer_for_stream.lock().unwrap();
                            buf.extend(mono_frame.iter().copied());
                            while buf.len() > VAD_BUFFER_MAX_SAMPLES {
                                buf.pop_front();
                            }
                        }
                        if is_recording_vad.load(Ordering::Acquire) {
                            audio_buffer_vad
                                .lock()
                                .unwrap()
                                .extend_from_slice(&mono_frame);
                            let levels = calculate_audio_levels(&mono_frame);
                            let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                            if should_emit_audio_levels_throttled(
                                &mut *last_sent,
                                memo_audio_levels_interval_ms(),
                            ) {
                                let json = json!(levels).to_string();
                                println_ui_flush!("AUDIO_LEVELS:{}", json);
                            }
                        }
                    },
                    |err| eprintln!("VAD stream error: {}", err),
                    None,
                ),
                _ => {
                    eprintln!("VAD: unsupported sample format");
                    return;
                }
            };
            if let Ok(stream) = stream_result {
                let _ = stream.play();
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(3600));
                }
            }
        });

        // Thread 2: VAD polling — RMS, state machine, send StartRecording/StopRecording
        std::thread::spawn(move || {
            let mut state = "idle"; // "idle" | "speech"
            let mut speech_above_ms: u64 = 0;
            let mut silence_below_ms: u64 = 0;
            let poll_duration = std::time::Duration::from_millis(vad_poll_interval_ms);

            loop {
                std::thread::sleep(poll_duration);
                let rms = {
                    let buf = vad_buffer_for_poll.lock().unwrap();
                    let len = buf.len();
                    if len >= samples_per_poll {
                        let start = len - samples_per_poll;
                        let slice: Vec<i16> = buf.range(start..).copied().collect();
                        compute_rms(&slice)
                    } else {
                        0.0
                    }
                };

                match state {
                    "idle" => {
                        if rms > vad_speech_threshold {
                            speech_above_ms += vad_poll_interval_ms;
                            if speech_above_ms >= vad_speech_start_ms {
                                state = "speech";
                                speech_above_ms = 0;
                                let _ = tx_vad.send(KeyEvent::StartRecording);
                            }
                        } else {
                            speech_above_ms = 0;
                        }
                    }
                    "speech" => {
                        if rms < vad_silence_threshold {
                            silence_below_ms += vad_poll_interval_ms;
                            if silence_below_ms >= vad_silence_ms {
                                state = "idle";
                                silence_below_ms = 0;
                                let _ = tx_vad.send(KeyEvent::StopRecording);
                            }
                        } else {
                            silence_below_ms = 0;
                        }
                    }
                    _ => {}
                }
            }
        });

        println!("\nTrigger: VAD (Radio mode)");
        println!("Speak to start recording, silence to transcribe.\n");
    } else {
        // System mode: keyboard hotkey trigger
        let trigger_pressed = Arc::new(AtomicBool::new(false));
        let control_pressed = Arc::new(AtomicBool::new(false));
        let lock_toggle_processed = Arc::new(AtomicBool::new(false));

        let trigger_pressed_clone = trigger_pressed.clone();
        let control_pressed_clone = control_pressed.clone();
        let lock_toggle_processed_clone = lock_toggle_processed.clone();
        let is_locked_listener = is_locked.clone();

        let trigger_key_for_listener = trigger_key;
        let tx_keyboard = tx.clone();
        std::thread::spawn(move || {
            listen(move |event: Event| {
                match event.event_type {
                    EventType::KeyPress(key) if key == trigger_key_for_listener => {
                        trigger_pressed_clone.store(true, Ordering::Release);

                        if control_pressed_clone.load(Ordering::Acquire) {
                            if !lock_toggle_processed_clone.swap(true, Ordering::Acquire) {
                                let _ = tx_keyboard.send(KeyEvent::ToggleLock);
                            }
                        } else {
                            let _ = tx_keyboard.send(KeyEvent::StartRecording);
                        }
                    }
                    EventType::KeyRelease(key) if key == trigger_key_for_listener => {
                        trigger_pressed_clone.store(false, Ordering::Release);
                        lock_toggle_processed_clone.store(false, Ordering::Release);

                        if !is_locked_listener.load(Ordering::Acquire) {
                            let _ = tx_keyboard.send(KeyEvent::StopRecording);
                        }
                    }
                    EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => {
                        control_pressed_clone.store(true, Ordering::Release);

                        if trigger_pressed_clone.load(Ordering::Acquire) {
                            if !lock_toggle_processed_clone.swap(true, Ordering::Acquire) {
                                let _ = tx_keyboard.send(KeyEvent::ToggleLock);
                            }
                        }
                    }
                    EventType::KeyRelease(Key::ControlLeft) | EventType::KeyRelease(Key::ControlRight) => {
                        control_pressed_clone.store(false, Ordering::Release);
                        lock_toggle_processed_clone.store(false, Ordering::Release);
                    }
                    _ => {}
                }
            }).ok();
        });

        println!("\nTrigger: Function key (or BLE device button)");
        println!("Press and hold to record, release to transcribe.");
        println!("Lock: Function+Control to toggle lock (keeps recording on)\n");
    }
    
    // Vocabulary storage for voice commands
    #[derive(Clone)]
    struct Vocabulary {
        app_names: Vec<String>,
        app_commands: HashMap<String, Vec<String>>,
        global_commands: Vec<String>,
    }
    
    let vocabulary = Arc::new(Mutex::new(Vocabulary {
        app_names: Vec::new(),
        app_commands: HashMap::new(),
        global_commands: Vec::new(),
    }));
    
    // Build prompt with only the active app's commands (+ global) to avoid hallucination
    let build_prompt = |app_name: String, window_title: String, vocab: &Vocabulary| -> Option<String> {
        let mut parts = Vec::new();
        
        if !app_name.is_empty() && app_name != "Unknown" {
            if !window_title.is_empty() {
                parts.push(format!("You are transcribing for {}. The current window is: {}.", app_name, window_title));
            } else {
                parts.push(format!("You are transcribing for {}.", app_name));
            }
        }
        
        if !vocab.app_names.is_empty() {
            parts.push(format!("Voice commands: open {}.", vocab.app_names.join(", ")));
        }
        
        let mut active_cmds: Vec<&str> = vocab.global_commands.iter().map(|s| s.as_str()).collect();
        if !app_name.is_empty() {
            let key = app_name.to_lowercase();
            if let Some(cmds) = vocab.app_commands.get(&key) {
                active_cmds.extend(cmds.iter().map(|s| s.as_str()));
            }
        }
        if !active_cmds.is_empty() {
            parts.push(format!("Commands: {}.", active_cmds.join(", ")));
        }
        
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };
    
    // Spawn thread to read commands from stdin
    let press_enter_clone = press_enter_after_paste.clone();
    let input_source = Arc::new(Mutex::new(String::from("system")));
    let input_source_clone = input_source.clone();
    let vocabulary_clone = vocabulary.clone();
    std::thread::spawn(move || {
        use std::io::{self, BufRead};
        let stdin = io::stdin();
        for line in stdin.lock().lines() {
            if let Ok(cmd) = line {
                if let Some(value) = cmd.strip_prefix("ENTER:") {
                    let enable = value.trim() == "1" || value.trim().eq_ignore_ascii_case("true");
                    press_enter_clone.store(enable, Ordering::Release);
                    eprintln!("MIC: Press-Enter-after-paste: {}", enable);
                } else if let Some(value) = cmd.strip_prefix("INPUT_SOURCE:") {
                    let source = value.trim().to_lowercase();
                    if source == "system" || source == "ble" {
                        let source_clone = source.clone();
                        *input_source_clone.lock().unwrap() = source;
                        eprintln!("MIC: Input source changed to: {}", source_clone);
                        // Note: INPUT_SOURCE changes are handled but require process restart in current architecture
                        // Future: dynamic source switching
                    }
                } else if cmd.starts_with("SCAN_START:") || cmd.starts_with("SCAN_STOP") ||
                          cmd.starts_with("CONNECT:") || cmd.starts_with("CONNECT_UID:") ||
                          cmd.trim() == "DISCONNECT" {
                    // BLE commands - only work in BLE mode, log them here
                    eprintln!("MIC: Received BLE command (requires BLE mode): {}", cmd);
                } else if let Some(value) = cmd.strip_prefix("VOCAB:") {
                    if let Ok(vocab_json) = serde_json::from_str::<serde_json::Value>(value.trim()) {
                        let mut vocab = vocabulary_clone.lock().unwrap();
                        vocab.app_names = vocab_json.get("appNames")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.global_commands = vocab_json.get("globalCommands")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.app_commands.clear();
                        if let Some(obj) = vocab_json.get("appCommands").and_then(|v| v.as_object()) {
                            for (key, val) in obj {
                                if let Some(arr) = val.as_array() {
                                    let cmds: Vec<String> = arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                                    vocab.app_commands.insert(key.clone(), cmds);
                                }
                            }
                        }
                        eprintln!("MIC: Vocabulary updated: {} app names, {} apps with commands, {} global commands",
                            vocab.app_names.len(), vocab.app_commands.len(), vocab.global_commands.len());
                    } else {
                        eprintln!("MIC: Failed to parse VOCAB command");
                    }
                }
            }
        }
    });

    loop {
        match rx.recv() {
            Ok(KeyEvent::StartRecording) => {
                if is_recording_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    println_ui_flush!("🎤 Recording...");
                    audio_buffer_clone.lock().unwrap().clear();

                    if streaming_enabled {
                        segment_results_clone.lock().unwrap().clear();
                        *segment_boundary_clone.lock().unwrap() = 0;
                        *last_segment_text_clone.lock().unwrap() = None;
                        segmenter_active_clone.store(true, Ordering::Release);

                        let buf_seg = audio_buffer_clone.clone();
                        let boundary_seg = segment_boundary_clone.clone();
                        let active_seg = segmenter_active_clone.clone();
                        let results_seg = segment_results_clone.clone();
                        let engine_seg = engine_clone.clone();
                        let vocab_seg = vocabulary.clone();
                        let prev_text_seg = last_segment_text_clone.clone();
                        let sr = sample_rate;
                        std::thread::spawn(move || {
                            let poll_ms: u64 = 50;
                            let samples_per_poll = (sr as u64 * poll_ms / 1000) as usize;
                            let mut silence_count_ms: u64 = 0;

                            loop {
                                std::thread::sleep(std::time::Duration::from_millis(poll_ms));
                                if !active_seg.load(Ordering::Acquire) {
                                    break;
                                }

                                let buf = buf_seg.lock().unwrap();
                                let current_len = buf.len();
                                let last_boundary = *boundary_seg.lock().unwrap();
                                let segment_samples = current_len.saturating_sub(last_boundary);
                                let segment_dur_ms = (segment_samples as u64 * 1000) / sr as u64;

                                if segment_dur_ms < seg_min_duration_ms {
                                    silence_count_ms = 0;
                                    drop(buf);
                                    continue;
                                }

                                let rms_window = samples_per_poll.min(segment_samples);
                                if rms_window == 0 { drop(buf); continue; }
                                let rms = compute_rms(&buf[current_len - rms_window..current_len]);

                                if rms < seg_silence_threshold {
                                    silence_count_ms += poll_ms;
                                } else {
                                    silence_count_ms = 0;
                                }

                                let should_segment = silence_count_ms >= seg_silence_ms
                                    || segment_dur_ms >= seg_max_duration_ms;

                                if should_segment {
                                    let segment_audio: Vec<i16> = buf[last_boundary..current_len].to_vec();
                                    *boundary_seg.lock().unwrap() = current_len;
                                    drop(buf);
                                    silence_count_ms = 0;

                                    let eng = engine_seg.clone();
                                    let res = results_seg.clone();
                                    let voc = vocab_seg.clone();
                                    let prev_text = prev_text_seg.clone();
                                    std::thread::spawn(move || {
                                        let mut eng = eng.lock().unwrap();
                                        let (app_name, window_title) = app_detection::get_application_context();
                                        let vocab = voc.lock().unwrap();
                                        let mut prompt = build_prompt(app_name, window_title, &vocab);
                                        if let Some(ref prev) = *prev_text.lock().unwrap() {
                                            let context = if prev.len() > 200 { &prev[prev.len()-200..] } else { prev.as_str() };
                                            prompt = Some(match prompt {
                                                Some(p) => format!("{} {}", p, context),
                                                None => context.to_string(),
                                            });
                                        }
                                        eng.set_prompt(prompt);
                                        match eng.transcribe(&segment_audio) {
                                            Ok(text) if !text.trim().is_empty() => {
                                                let trimmed = text.trim().to_string();
                                                *prev_text.lock().unwrap() = Some(trimmed.clone());
                                                res.lock().unwrap().push(trimmed);
                                                eprintln!("[Streaming] Segment transcribed ({} chars)", text.len());
                                            }
                                            Ok(_) => eprintln!("[Streaming] Segment had no speech"),
                                            Err(e) => eprintln!("[Streaming] Segment error: {}", e),
                                        }
                                    });
                                } else {
                                    drop(buf);
                                }
                            }
                        });
                    }

                    if !use_vad_trigger {
                        let buffer = audio_buffer_clone.clone();
                        let is_recording_for_audio = is_recording_clone.clone();
                        let last_audio_level_sent = Arc::new(Mutex::new(None::<Instant>));
                        let last_audio_level_sent_clone = last_audio_level_sent.clone();
                        let stream_config = config_clone.clone().into();
                        let stream_result = match config_clone.sample_format() {
                        cpal::SampleFormat::I16 => {
                            device_clone.build_input_stream(
                                &stream_config,
                                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                    let mut b = buffer.lock().unwrap();
                                    extend_buffer_mono_i16(&mut *b, data, stream_ch);

                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = audio_levels_interleaved_i16(data, stream_ch);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        if should_emit_audio_levels_throttled(
                                            &mut *last_sent,
                                            memo_audio_levels_interval_ms(),
                                        ) {
                                            let json = json!(levels).to_string();
                                            println_ui_flush!("AUDIO_LEVELS:{}", json);
                                        }
                                    }
                                },
                                |err| eprintln!("Audio error: {}", err),
                                None,
                            )
                        }
                        cpal::SampleFormat::F32 => {
                            device_clone.build_input_stream(
                                &stream_config,
                                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                    let mut buf = buffer.lock().unwrap();
                                    extend_buffer_mono_f32(&mut *buf, data, stream_ch);

                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = audio_levels_interleaved_f32(data, stream_ch);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        if should_emit_audio_levels_throttled(
                                            &mut *last_sent,
                                            memo_audio_levels_interval_ms(),
                                        ) {
                                            let json = json!(levels).to_string();
                                            println_ui_flush!("AUDIO_LEVELS:{}", json);
                                        }
                                    }
                                },
                                |err| eprintln!("Audio error: {}", err),
                                None,
                            )
                        }
                        cpal::SampleFormat::U16 => {
                            device_clone.build_input_stream(
                                &stream_config,
                                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                                    let mut buf = buffer.lock().unwrap();
                                    extend_buffer_mono_u16(&mut *buf, data, stream_ch);

                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = audio_levels_interleaved_u16(data, stream_ch);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        if should_emit_audio_levels_throttled(
                                            &mut *last_sent,
                                            memo_audio_levels_interval_ms(),
                                        ) {
                                            let json = json!(levels).to_string();
                                            println_ui_flush!("AUDIO_LEVELS:{}", json);
                                        }
                                    }
                                },
                                |err| eprintln!("Audio error: {}", err),
                                None,
                            )
                        }
                        _ => {
                            eprintln!("Unsupported format");
                            continue;
                        }
                    };
                    
                        if let Ok(stream) = stream_result {
                            stream.play().ok();
                            *recording_stream_clone.lock().unwrap() = Some(stream);
                        } else {
                            is_recording_clone.store(false, Ordering::SeqCst);
                        }
                    }
                }
            }
            Ok(KeyEvent::StopRecording) => {
                if is_recording_clone.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    segmenter_active_clone.store(false, Ordering::Release);

                    if !use_vad_trigger {
                        recording_stream_clone.lock().unwrap().take();
                    }
                    
                    let samples = {
                        let mut buf = audio_buffer_clone.lock().unwrap();
                        std::mem::take(&mut *buf)
                    };

                    let streaming_boundary = if streaming_enabled {
                        *segment_boundary_clone.lock().unwrap()
                    } else { 0 };
                    
                    if !samples.is_empty() {
                        // Encode audio to OPUS for saving (only for 16kHz, resample if needed)
                        let samples_for_encoding = samples.clone();
                        let sample_rate_for_encoding = sample_rate;
                        let audio_duration = samples.len() as f32 / sample_rate as f32;
                        
                        // Encode in a separate thread (only if 16kHz, otherwise skip)
                        if sample_rate_for_encoding == 16000 {
                            std::thread::spawn(move || {
                                use opus_decoder::OpusEncoder;
                                #[cfg(feature = "binary")]
                                use base64::{Engine as _, engine::general_purpose::STANDARD};
                                
                                match OpusEncoder::new(16000, 20) {
                                    Ok(mut encoder_for_thread) => {
                                        match encoder_for_thread.encode_buffer(&samples_for_encoding) {
                                            Ok(opus_data) => {
                                                #[cfg(feature = "binary")]
                                                {
                                                    let base64_data = STANDARD.encode(&opus_data);
                                                    println!("AUDIO_DATA:{}", base64_data);
                                                    println!("AUDIO_DURATION:{:.2}", audio_duration);
                                                    
                                                    // Also output WAV data for easy playback
                                                    let sample_rate = 16000u32;
                                                    let channels = 1u16;
                                                    let bits_per_sample = 16u16;
                                                    let pcm_data_len = samples_for_encoding.len() * 2;
                                                    let wav_size = 44 + pcm_data_len;
                                                    
                                                    let mut wav_data = Vec::with_capacity(wav_size);
                                                    wav_data.extend_from_slice(b"RIFF");
                                                    wav_data.extend_from_slice(&(36u32 + pcm_data_len as u32).to_le_bytes());
                                                    wav_data.extend_from_slice(b"WAVE");
                                                    wav_data.extend_from_slice(b"fmt ");
                                                    wav_data.extend_from_slice(&16u32.to_le_bytes());
                                                    wav_data.extend_from_slice(&1u16.to_le_bytes());
                                                    wav_data.extend_from_slice(&channels.to_le_bytes());
                                                    wav_data.extend_from_slice(&sample_rate.to_le_bytes());
                                                    wav_data.extend_from_slice(&(sample_rate as u32 * channels as u32 * (bits_per_sample as u32 / 8)).to_le_bytes());
                                                    wav_data.extend_from_slice(&(channels * (bits_per_sample / 8)).to_le_bytes());
                                                    wav_data.extend_from_slice(&bits_per_sample.to_le_bytes());
                                                    wav_data.extend_from_slice(b"data");
                                                    wav_data.extend_from_slice(&(pcm_data_len as u32).to_le_bytes());
                                                    for &sample in &samples_for_encoding {
                                                        wav_data.extend_from_slice(&sample.to_le_bytes());
                                                    }
                                                    
                                                    let wav_base64 = STANDARD.encode(&wav_data);
                                                    println!("AUDIO_WAV:{}", wav_base64);
                                                }
                                            }
                                            Err(e) => {
                                                eprintln!("Failed to encode audio: {}", e);
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to create Opus encoder: {}", e);
                                    }
                                }
                            });
                        }
                        
                        // Spawn transcription thread immediately for fastest response
                        let engine_for_thread = engine_clone.clone();
                        let perf_history = performance_history_clone.clone();
                        let press_enter_clone = press_enter_after_paste.clone();
                        let no_inject_clone = no_inject_flag.clone();
                        let vocabulary_for_thread = vocabulary.clone();
                        let segment_results_for_thread = segment_results_clone.clone();
                        let last_seg_text_for_thread = last_segment_text_clone.clone();
                        let sample_count = samples.len();
                        let audio_duration = sample_count as f32 / sample_rate as f32;
                        let start_time = Instant::now();
                        std::thread::spawn(move || {
                            println_ui_flush!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                            println!("🔄 Transcribing...");
                            let mut eng = engine_for_thread.lock().unwrap();
                            
                            // Capture application context and vocabulary before transcribing
                            let (app_name, window_title) = app_detection::get_application_context();
                            let vocab = vocabulary_for_thread.lock().unwrap();
                            let mut prompt = build_prompt(app_name, window_title, &vocab);
                            if streaming_boundary > 0 {
                                if let Some(ref prev) = *last_seg_text_for_thread.lock().unwrap() {
                                    let context = if prev.len() > 200 { &prev[prev.len()-200..] } else { prev.as_str() };
                                    prompt = Some(match prompt {
                                        Some(p) => format!("{} {}", p, context),
                                        None => context.to_string(),
                                    });
                                }
                            }
                            eng.set_prompt(prompt);

                            let accumulated_segments = if streaming_boundary > 0 {
                                std::mem::take(&mut *segment_results_for_thread.lock().unwrap())
                            } else { Vec::new() };
                            let pre_processed_count = accumulated_segments.len();

                            let transcribe_start = Instant::now();
                            let transcribe_result = if streaming_boundary > 0 {
                                let final_text = if streaming_boundary < samples.len()
                                    && samples.len() - streaming_boundary >= sample_rate as usize
                                {
                                    eng.transcribe(&samples[streaming_boundary..]).unwrap_or_default()
                                } else if streaming_boundary < samples.len() {
                                    String::new()
                                } else {
                                    String::new()
                                };
                                let mut parts = accumulated_segments;
                                if !final_text.trim().is_empty() {
                                    parts.push(final_text.trim().to_string());
                                }
                                let combined = join_segments(&parts);
                                if combined.trim().is_empty() {
                                    eng.transcribe(&samples)
                                } else {
                                    if pre_processed_count > 0 {
                                        eprintln!("[Streaming] {} segments pre-processed, final segment transcribed", pre_processed_count);
                                    }
                                    Ok(combined)
                                }
                            } else {
                                eng.transcribe(&samples)
                            };
                            match transcribe_result {
                                Ok(text) => {
                                    let transcribe_time = transcribe_start.elapsed();
                                    let realtime_factor = audio_duration / transcribe_time.as_secs_f32();
                                    
                                    // Update performance history
                                    {
                                        let mut history = perf_history.lock().unwrap();
                                        history.push_back((audio_duration, realtime_factor));
                                        if history.len() > 10 {
                                            history.pop_front();
                                        }
                                    }
                                    
                                    // Calculate rate of increase
                                    let rate_info = {
                                        let history = perf_history.lock().unwrap();
                                        let history_vec: Vec<(f32, f32)> = history.iter().copied().collect();
                                        if history_vec.len() >= 2 {
                                            if let Some(rate) = calculate_rate_of_increase(&history_vec) {
                                                let predicted_30s = history_vec.last().unwrap().1 + rate * (30.0 - history_vec.last().unwrap().0);
                                                let predicted_60s = history_vec.last().unwrap().1 + rate * (60.0 - history_vec.last().unwrap().0);
                                                Some((rate, predicted_30s, predicted_60s))
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    };
                                    
                                    if text.trim().is_empty() {
                                        println!("📝 (no speech detected)");
                                        println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime)", 
                                                transcribe_time.as_secs_f32() * 1000.0, realtime_factor);
                                        if let Some((rate, pred_30, pred_60)) = rate_info {
                                            println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                        } else {
                                            println!();
                                        }
                                    } else {
                                        // Capture application context (already captured before transcription)
                                        let (app_name, window_title) = app_detection::get_application_context();
                                        
                                        // Process text to strip periods from short phrases
                                        let processed_text = strip_leading_dash_space(&strip_trailing_signoffs(&strip_periods_from_short_phrases(&text)));
                                        
                                        // Output FINAL: JSON for Electron app integration
                                        let json_output = json!({
                                            "rawTranscript": text,
                                            "processedText": processed_text,
                                            "wasProcessedByLLM": false,
                                            "appContext": {
                                                "appName": app_name,
                                                "windowTitle": window_title
                                            }
                                        });
                                        println!("FINAL: {}", json_output);
                                        
                                        // Only inject if not in Electron mode
                                        if !no_inject_clone.load(Ordering::Acquire) {
                                            // Inject first for fastest response time
                                            let inject_start = Instant::now();
                                            let press_enter = press_enter_clone.load(Ordering::Acquire);
                                            match inject_text(&processed_text, press_enter) {
                                                Ok(_) => {
                                                    let inject_time = inject_start.elapsed();
                                                    let total_time = start_time.elapsed();
                                                    println!("📝 {}", text);
                                                    println!("✅ Injected");
                                                    println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Injection: {:.2}ms | Total: {:.2}ms",
                                                            transcribe_time.as_secs_f32() * 1000.0, 
                                                            realtime_factor,
                                                            inject_time.as_secs_f32() * 1000.0,
                                                            total_time.as_secs_f32() * 1000.0);
                                                    if let Some((rate, pred_30, pred_60)) = rate_info {
                                                        println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                    } else {
                                                        println!();
                                                    }
                                                }
                                                Err(e) => {
                                                    let inject_time = inject_start.elapsed();
                                                    let total_time = start_time.elapsed();
                                                    println!("📝 {}", text);
                                                    eprintln!("❌ Injection failed: {}", e);
                                                    println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Injection: {:.2}ms | Total: {:.2}ms",
                                                            transcribe_time.as_secs_f32() * 1000.0, 
                                                            realtime_factor,
                                                            inject_time.as_secs_f32() * 1000.0,
                                                            total_time.as_secs_f32() * 1000.0);
                                                    if let Some((rate, pred_30, pred_60)) = rate_info {
                                                        println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                    } else {
                                                        println!();
                                                    }
                                                }
                                            }
                                        } else {
                                            let total_time = start_time.elapsed();
                                            println!("📝 {}", text);
                                            println!("⏭️  Injection skipped (Electron mode)");
                                            println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Total: {:.2}ms",
                                                    transcribe_time.as_secs_f32() * 1000.0, 
                                                    realtime_factor,
                                                    total_time.as_secs_f32() * 1000.0);
                                            if let Some((rate, pred_30, pred_60)) = rate_info {
                                                println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                            } else {
                                                println!();
                                            }
                                        }
                                    }
                                }
                                Err(e) => {
                                    let total_time = start_time.elapsed();
                                    eprintln!("❌ Error: {}", e);
                                    println!("⏱️  Total time: {:.2}ms\n", total_time.as_secs_f32() * 1000.0);
                                }
                            }
                        });
                    }
                }
            }
            Ok(KeyEvent::ToggleLock) => {
                let was_locked = is_locked_clone.load(Ordering::Acquire);
                let now_locked = !was_locked;
                is_locked_clone.store(now_locked, Ordering::Release);
                
                if now_locked {
                    // Locking: ensure recording is on
                    println!("🔒 Locked - recording will continue until unlocked");
                    if !is_recording_clone.load(Ordering::Acquire) {
                        // Start recording if not already recording
                        // Manually trigger start recording logic
                        if is_recording_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            println_ui_flush!("🎤 Recording...");
                            audio_buffer_clone.lock().unwrap().clear();

                            if streaming_enabled {
                                segment_results_clone.lock().unwrap().clear();
                                *segment_boundary_clone.lock().unwrap() = 0;
                                *last_segment_text_clone.lock().unwrap() = None;
                                segmenter_active_clone.store(true, Ordering::Release);

                                let buf_seg = audio_buffer_clone.clone();
                                let boundary_seg = segment_boundary_clone.clone();
                                let active_seg = segmenter_active_clone.clone();
                                let results_seg = segment_results_clone.clone();
                                let engine_seg = engine_clone.clone();
                                let vocab_seg = vocabulary.clone();
                                let prev_text_seg = last_segment_text_clone.clone();
                                let sr = sample_rate;
                                std::thread::spawn(move || {
                                    let poll_ms: u64 = 50;
                                    let samples_per_poll = (sr as u64 * poll_ms / 1000) as usize;
                                    let mut silence_count_ms: u64 = 0;
                                    loop {
                                        std::thread::sleep(std::time::Duration::from_millis(poll_ms));
                                        if !active_seg.load(Ordering::Acquire) { break; }
                                        let buf = buf_seg.lock().unwrap();
                                        let current_len = buf.len();
                                        let last_boundary = *boundary_seg.lock().unwrap();
                                        let segment_samples = current_len.saturating_sub(last_boundary);
                                        let segment_dur_ms = (segment_samples as u64 * 1000) / sr as u64;
                                        if segment_dur_ms < seg_min_duration_ms {
                                            silence_count_ms = 0; drop(buf); continue;
                                        }
                                        let rms_window = samples_per_poll.min(segment_samples);
                                        if rms_window == 0 { drop(buf); continue; }
                                        let rms = compute_rms(&buf[current_len - rms_window..current_len]);
                                        if rms < seg_silence_threshold { silence_count_ms += poll_ms; } else { silence_count_ms = 0; }
                                        let should_segment = silence_count_ms >= seg_silence_ms || segment_dur_ms >= seg_max_duration_ms;
                                        if should_segment {
                                            let segment_audio: Vec<i16> = buf[last_boundary..current_len].to_vec();
                                            *boundary_seg.lock().unwrap() = current_len;
                                            drop(buf);
                                            silence_count_ms = 0;
                                            let eng = engine_seg.clone();
                                            let res = results_seg.clone();
                                            let voc = vocab_seg.clone();
                                            let prev_text = prev_text_seg.clone();
                                            std::thread::spawn(move || {
                                                let mut eng = eng.lock().unwrap();
                                                let (app_name, window_title) = app_detection::get_application_context();
                                                let vocab = voc.lock().unwrap();
                                                let mut prompt = build_prompt(app_name, window_title, &vocab);
                                                if let Some(ref prev) = *prev_text.lock().unwrap() {
                                                    let context = if prev.len() > 200 { &prev[prev.len()-200..] } else { prev.as_str() };
                                                    prompt = Some(match prompt {
                                                        Some(p) => format!("{} {}", p, context),
                                                        None => context.to_string(),
                                                    });
                                                }
                                                eng.set_prompt(prompt);
                                                match eng.transcribe(&segment_audio) {
                                                    Ok(text) if !text.trim().is_empty() => {
                                                        let trimmed = text.trim().to_string();
                                                        *prev_text.lock().unwrap() = Some(trimmed.clone());
                                                        res.lock().unwrap().push(trimmed);
                                                        eprintln!("[Streaming] Segment transcribed ({} chars)", text.len());
                                                    }
                                                    Ok(_) => eprintln!("[Streaming] Segment had no speech"),
                                                    Err(e) => eprintln!("[Streaming] Segment error: {}", e),
                                                }
                                            });
                                        } else { drop(buf); }
                                    }
                                });
                            }

                            let buffer = audio_buffer_clone.clone();
                            let is_recording_for_audio_lock = is_recording_clone.clone();
                            let last_audio_level_sent_lock = Arc::new(Mutex::new(None::<Instant>));
                            let last_audio_level_sent_lock_clone = last_audio_level_sent_lock.clone();
                            let stream_config = config_clone.clone().into();
                            let stream_result = match config_clone.sample_format() {
                                cpal::SampleFormat::I16 => {
                                    device_clone.build_input_stream(
                                        &stream_config,
                                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                            let mut b = buffer.lock().unwrap();
                                            extend_buffer_mono_i16(&mut *b, data, stream_ch);

                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = audio_levels_interleaved_i16(data, stream_ch);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if should_emit_audio_levels_throttled(
                                                    &mut *last_sent,
                                                    memo_audio_levels_interval_ms(),
                                                ) {
                                                    let json = json!(levels).to_string();
                                                    println_ui_flush!("AUDIO_LEVELS:{}", json);
                                                }
                                            }
                                        },
                                        |err| eprintln!("Audio error: {}", err),
                                        None,
                                    )
                                }
                                cpal::SampleFormat::F32 => {
                                    device_clone.build_input_stream(
                                        &stream_config,
                                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                            let mut buf = buffer.lock().unwrap();
                                            extend_buffer_mono_f32(&mut *buf, data, stream_ch);

                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = audio_levels_interleaved_f32(data, stream_ch);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if should_emit_audio_levels_throttled(
                                                    &mut *last_sent,
                                                    memo_audio_levels_interval_ms(),
                                                ) {
                                                    let json = json!(levels).to_string();
                                                    println_ui_flush!("AUDIO_LEVELS:{}", json);
                                                }
                                            }
                                        },
                                        |err| eprintln!("Audio error: {}", err),
                                        None,
                                    )
                                }
                                cpal::SampleFormat::U16 => {
                                    device_clone.build_input_stream(
                                        &stream_config,
                                        move |data: &[u16], _: &cpal::InputCallbackInfo| {
                                            let mut buf = buffer.lock().unwrap();
                                            extend_buffer_mono_u16(&mut *buf, data, stream_ch);

                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = audio_levels_interleaved_u16(data, stream_ch);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if should_emit_audio_levels_throttled(
                                                    &mut *last_sent,
                                                    memo_audio_levels_interval_ms(),
                                                ) {
                                                    let json = json!(levels).to_string();
                                                    println_ui_flush!("AUDIO_LEVELS:{}", json);
                                                }
                                            }
                                        },
                                        |err| eprintln!("Audio error: {}", err),
                                        None,
                                    )
                                }
                                _ => {
                                    eprintln!("Unsupported format");
                                    continue;
                                }
                            };
                            
                            if let Ok(stream) = stream_result {
                                stream.play().ok();
                                *recording_stream_clone.lock().unwrap() = Some(stream);
                            } else {
                                is_recording_clone.store(false, Ordering::SeqCst);
                            }
                        }
                    }
                } else {
                    // Unlocking: stop recording
                    println!("🔓 Unlocked");
                    if is_recording_clone.load(Ordering::Acquire) {
                        // Manually trigger stop recording logic
                        if is_recording_clone.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                            segmenter_active_clone.store(false, Ordering::Release);
                            recording_stream_clone.lock().unwrap().take();
                            
                            let samples = {
                                let mut buf = audio_buffer_clone.lock().unwrap();
                                std::mem::take(&mut *buf)
                            };

                            let streaming_boundary = if streaming_enabled {
                                *segment_boundary_clone.lock().unwrap()
                            } else { 0 };
                            
                            if !samples.is_empty() {
                                // Spawn transcription thread immediately for fastest response
                                let engine_for_thread = engine_clone.clone();
                                let perf_history = performance_history_clone.clone();
                                let press_enter_clone = press_enter_after_paste.clone();
                                let no_inject_clone = no_inject_flag.clone();
                                let vocabulary_for_thread = vocabulary.clone();
                                let segment_results_for_thread = segment_results_clone.clone();
                                let last_seg_text_for_thread = last_segment_text_clone.clone();
                                let sample_count = samples.len();
                                let audio_duration = sample_count as f32 / sample_rate as f32;
                                let start_time = Instant::now();
                                std::thread::spawn(move || {
                                    println_ui_flush!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                                    println!("🔄 Transcribing...");
                                    let mut eng = engine_for_thread.lock().unwrap();
                                    
                                    // Capture application context and vocabulary before transcribing
                                    let (app_name, window_title) = app_detection::get_application_context();
                                    let vocab = vocabulary_for_thread.lock().unwrap();
                                    let mut prompt = build_prompt(app_name, window_title, &vocab);
                                    if streaming_boundary > 0 {
                                        if let Some(ref prev) = *last_seg_text_for_thread.lock().unwrap() {
                                            let context = if prev.len() > 200 { &prev[prev.len()-200..] } else { prev.as_str() };
                                            prompt = Some(match prompt {
                                                Some(p) => format!("{} {}", p, context),
                                                None => context.to_string(),
                                            });
                                        }
                                    }
                                    eng.set_prompt(prompt);

                                    let accumulated_segments = if streaming_boundary > 0 {
                                        std::mem::take(&mut *segment_results_for_thread.lock().unwrap())
                                    } else { Vec::new() };
                                    let pre_processed_count = accumulated_segments.len();

                                    let transcribe_start = Instant::now();
                                    let transcribe_result = if streaming_boundary > 0 {
                                        let final_text = if streaming_boundary < samples.len()
                                            && samples.len() - streaming_boundary >= sample_rate as usize
                                        {
                                            eng.transcribe(&samples[streaming_boundary..]).unwrap_or_default()
                                        } else if streaming_boundary < samples.len() {
                                            String::new()
                                        } else {
                                            String::new()
                                        };
                                        let mut parts = accumulated_segments;
                                        if !final_text.trim().is_empty() {
                                            parts.push(final_text.trim().to_string());
                                        }
                                        let combined = join_segments(&parts);
                                        if combined.trim().is_empty() {
                                            eng.transcribe(&samples)
                                        } else {
                                            if pre_processed_count > 0 {
                                                eprintln!("[Streaming] {} segments pre-processed, final segment transcribed", pre_processed_count);
                                            }
                                            Ok(combined)
                                        }
                                    } else {
                                        eng.transcribe(&samples)
                                    };
                                    match transcribe_result {
                                        Ok(text) => {
                                            let transcribe_time = transcribe_start.elapsed();
                                            let realtime_factor = audio_duration / transcribe_time.as_secs_f32();
                                            
                                            // Update performance history
                                            {
                                                let mut history = perf_history.lock().unwrap();
                                                history.push_back((audio_duration, realtime_factor));
                                                if history.len() > 10 {
                                                    history.pop_front();
                                                }
                                            }
                                            
                                            // Calculate rate of increase
                                            let rate_info = {
                                                let history = perf_history.lock().unwrap();
                                                let history_vec: Vec<(f32, f32)> = history.iter().copied().collect();
                                                if history_vec.len() >= 2 {
                                                    if let Some(rate) = calculate_rate_of_increase(&history_vec) {
                                                        let predicted_30s = history_vec.last().unwrap().1 + rate * (30.0 - history_vec.last().unwrap().0);
                                                        let predicted_60s = history_vec.last().unwrap().1 + rate * (60.0 - history_vec.last().unwrap().0);
                                                        Some((rate, predicted_30s, predicted_60s))
                                                    } else {
                                                        None
                                                    }
                                                } else {
                                                    None
                                                }
                                            };
                                            
                                            if text.trim().is_empty() {
                                                println!("📝 (no speech detected)");
                                                println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime)", 
                                                        transcribe_time.as_secs_f32() * 1000.0, realtime_factor);
                                                if let Some((rate, pred_30, pred_60)) = rate_info {
                                                    println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                } else {
                                                    println!();
                                                }
                                            } else {
                                                // Capture application context (already captured before transcription)
                                                let (app_name, window_title) = app_detection::get_application_context();
                                                
                                                // Process text to strip periods from short phrases
                                                let processed_text = strip_leading_dash_space(&strip_trailing_signoffs(&strip_periods_from_short_phrases(&text)));
                                                
                                                // Output FINAL: JSON for Electron app integration
                                                let json_output = json!({
                                                    "rawTranscript": text,
                                                    "processedText": processed_text,
                                                    "wasProcessedByLLM": false,
                                                    "appContext": {
                                                        "appName": app_name,
                                                        "windowTitle": window_title
                                                    }
                                                });
                                                println!("FINAL: {}", json_output);
                                                
                                                // Only inject if not in Electron mode
                                                if !no_inject_clone.load(Ordering::Acquire) {
                                                    // Inject first for fastest response time
                                                    let inject_start = Instant::now();
                                                    let press_enter = press_enter_clone.load(Ordering::Acquire);
                                                    match inject_text(&processed_text, press_enter) {
                                                        Ok(_) => {
                                                            let inject_time = inject_start.elapsed();
                                                            let total_time = start_time.elapsed();
                                                            println!("📝 {}", text);
                                                            println!("✅ Injected");
                                                            println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Injection: {:.2}ms | Total: {:.2}ms",
                                                                    transcribe_time.as_secs_f32() * 1000.0, 
                                                                    realtime_factor,
                                                                    inject_time.as_secs_f32() * 1000.0,
                                                                    total_time.as_secs_f32() * 1000.0);
                                                            if let Some((rate, pred_30, pred_60)) = rate_info {
                                                                println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                            } else {
                                                                println!();
                                                            }
                                                        }
                                                        Err(e) => {
                                                            let inject_time = inject_start.elapsed();
                                                            let total_time = start_time.elapsed();
                                                            println!("📝 {}", text);
                                                            eprintln!("❌ Injection failed: {}", e);
                                                            println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Injection: {:.2}ms | Total: {:.2}ms",
                                                                    transcribe_time.as_secs_f32() * 1000.0, 
                                                                    realtime_factor,
                                                                    inject_time.as_secs_f32() * 1000.0,
                                                                    total_time.as_secs_f32() * 1000.0);
                                                            if let Some((rate, pred_30, pred_60)) = rate_info {
                                                                println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                            } else {
                                                                println!();
                                                            }
                                                        }
                                                    }
                                                } else {
                                                    let total_time = start_time.elapsed();
                                                    println!("📝 {}", text);
                                                    println!("⏭️  Injection skipped (Electron mode)");
                                                    println!("⏱️  Transcription: {:.2}ms ({:.2}x realtime) | Total: {:.2}ms",
                                                            transcribe_time.as_secs_f32() * 1000.0, 
                                                            realtime_factor,
                                                            total_time.as_secs_f32() * 1000.0);
                                                    if let Some((rate, pred_30, pred_60)) = rate_info {
                                                        println!("📈 Rate: +{:.2}x per second | Predicted: {:.1}x @ 30s, {:.1}x @ 60s\n", rate, pred_30, pred_60);
                                                    } else {
                                                        println!();
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            let total_time = start_time.elapsed();
                                            eprintln!("❌ Error: {}", e);
                                            println!("⏱️  Total time: {:.2}ms\n", total_time.as_secs_f32() * 1000.0);
                                        }
                                    }
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {:?}", e);
                return Err(e.into());
            }
        }
    }
}

