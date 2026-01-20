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
use std::collections::VecDeque;
use serde_json::json;
#[cfg(feature = "binary")]
use log::debug;
mod app_detection;

#[cfg(feature = "binary")]
mod ble;
#[cfg(feature = "binary")]
mod opus_decoder;

/// Strip periods from phrases or sentences that are less than 4 words long
fn strip_periods_from_short_phrases(text: &str) -> String {
    // Split by common sentence delimiters (period, exclamation, question mark)
    let mut result = String::new();
    let mut current_phrase = String::new();
    
    for ch in text.chars() {
        if ch == '.' || ch == '!' || ch == '?' {
            // End of a phrase/sentence
            let phrase = current_phrase.trim();
            if !phrase.is_empty() {
                // Count words in the phrase
                let word_count = phrase.split_whitespace().count();
                
                if word_count < 4 {
                    // Strip the period for short phrases
                    result.push_str(&phrase);
                    // Don't add the period
                } else {
                    // Keep the period for longer phrases
                    result.push_str(&phrase);
                    result.push(ch);
                }
                result.push(' ');
            }
            current_phrase.clear();
        } else {
            current_phrase.push(ch);
        }
    }
    
    // Handle any remaining text (no trailing punctuation)
    if !current_phrase.trim().is_empty() {
        let phrase = current_phrase.trim();
        result.push_str(&phrase);
    }
    
    result.trim().to_string()
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

// Calculate audio levels for BLE audio with reduced sensitivity
// Uses a higher normalization threshold to make the waveform less reactive
fn calculate_audio_levels_ble(samples: &[i16]) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; 7];
    }
    
    // Calculate RMS (Root Mean Square) for audio level
    let sum_squares: i64 = samples.iter().map(|&s| (s as i64).pow(2)).sum();
    let rms = (sum_squares as f32 / samples.len() as f32).sqrt();
    
    // Normalize to 0-1 range with higher threshold for reduced sensitivity
    // Higher threshold = less sensitive (requires louder audio to reach full scale)
    const NORMALIZATION_THRESHOLD: f32 = 20000.0;  // Increased from 15000.0
    const GAIN_BOOST: f32 = 1.5;  // Reduced from 2.0
    let normalized = ((rms / NORMALIZATION_THRESHOLD) * GAIN_BOOST).min(1.0);
    
    // Apply exponential scaling for better visual response
    let scaled = normalized.powf(0.4);
    
    // Create 7 bands with symmetric weighting (center bars higher, edges taper down)
    let weights = vec![0.6, 0.8, 0.95, 1.0, 0.95, 0.8, 0.6];
    weights.into_iter()
        .map(|w| (scaled * w).min(1.0))
        .collect()
}
#[cfg(not(target_os = "macos"))]
use enigo::{Enigo, KeyboardControllable, Key as EnigoKey};

// Default trigger key (can be overridden via --hotkey argument)
const DEFAULT_TRIGGER_KEY: Key = Key::Function;

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
async fn run_ble_audio_mode(engine: Arc<Mutex<SttEngine>>) -> Result<(), Box<dyn std::error::Error>> {
    use ble::BleAudioReceiver;
    use opus_decoder::OpusDecoder;
    
    println!("Starting BLE audio mode...");
    
    // Initialize Opus decoder (preserved during reconnection)
    let mut decoder = OpusDecoder::new(16000, 20)?;
    
    // Initialize BLE receiver
    let mut ble_receiver = BleAudioReceiver::new().await?;
    
    // Get preferred device name from environment variable
    let preferred_device_name = std::env::var("MEMO_DEVICE_NAME")
        .ok()
        .filter(|s| !s.is_empty() && s.to_lowercase().starts_with("memo_"));
    
    if let Some(ref name) = preferred_device_name {
        println!("Using preferred device: {}", name);
    }
    
    // Reconnection state (preserved across reconnection attempts)
    let mut reconnect_attempts = 0u32;
    const RECONNECT_DELAY_BASE: u64 = 2; // seconds
    const RECONNECT_DELAY_MAX: u64 = 30; // seconds
    
    // Connect to device (initial connection)
    println!("Connecting to memo device...");
    if let Err(e) = ble_receiver.connect(preferred_device_name.as_deref()).await {
        eprintln!("Failed to connect to BLE device: {}", e);
        eprintln!("Falling back to system microphone...");
        // Don't exit - fall back to system mic mode instead
        // This allows the app to continue working even if BLE is unavailable
        return Err(e.into());
    }
    
    // Device name is already printed by ble_receiver.connect() via eprintln!
    println!("Connected! Waiting for button press to start recording...");
    
    // State that persists across reconnections (preserved during reconnection)
    let engine_clone = engine.clone();
    let performance_history: Arc<Mutex<VecDeque<(f32, f32)>>> = Arc::new(Mutex::new(VecDeque::with_capacity(10)));
    let press_enter_after_paste = Arc::new(AtomicBool::new(false));
    let is_recording = Arc::new(AtomicBool::new(false));
    let audio_buffer = Arc::new(Mutex::new(Vec::<i16>::new()));
    
    // Vocabulary storage for voice commands
    #[derive(Clone)]
    struct Vocabulary {
        apps: Vec<String>,
        commands: Vec<String>,
    }
    
    let vocabulary = Arc::new(Mutex::new(Vocabulary {
        apps: Vec::new(),
        commands: Vec::new(),
    }));
    
    // Helper function to build combined prompt with app context and vocabulary
    let build_prompt = |app_name: String, window_title: String, vocab: &Vocabulary| -> Option<String> {
        let mut parts = Vec::new();
        
        // Add app context
        if !app_name.is_empty() && app_name != "Unknown" {
            if !window_title.is_empty() {
                parts.push(format!("You are transcribing for {}. The current window is: {}.", app_name, window_title));
            } else {
                parts.push(format!("You are transcribing for {}.", app_name));
            }
        }
        
        // Add vocabulary for voice commands
        if !vocab.apps.is_empty() || !vocab.commands.is_empty() {
            let mut vocab_parts = Vec::new();
            if !vocab.apps.is_empty() {
                vocab_parts.push(format!("Voice commands: open {}.", vocab.apps.join(", ")));
            }
            if !vocab.commands.is_empty() {
                vocab_parts.push(format!("Commands: {}.", vocab.commands.join(", ")));
            }
            if !vocab_parts.is_empty() {
                parts.push(vocab_parts.join(" "));
            }
        }
        
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };
    
    // Spawn thread to read commands from stdin
    let press_enter_clone = press_enter_after_paste.clone();
    let input_source = Arc::new(Mutex::new(String::from("ble")));
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
                    }
                } else if let Some(value) = cmd.strip_prefix("VOCAB:") {
                    // Parse vocabulary JSON: {"apps": [...], "commands": [...]}
                    if let Ok(vocab_json) = serde_json::from_str::<serde_json::Value>(value.trim()) {
                        let mut vocab = vocabulary_clone.lock().unwrap();
                        vocab.apps = vocab_json.get("apps")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.commands = vocab_json.get("commands")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        eprintln!("MIC: Vocabulary updated: {} apps, {} commands", vocab.apps.len(), vocab.commands.len());
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
    let last_audio_level_sent = Arc::new(Mutex::new(Instant::now()));
    let last_audio_level_sent_clone = last_audio_level_sent.clone();
    
    use ble::NotificationResult;
    use futures::StreamExt;
    use tokio::time::timeout;
    
    // Helper function for reconnection logic
    async fn attempt_reconnect(
        ble_receiver: &mut BleAudioReceiver,
        reconnect_attempts: &mut u32,
        preferred_device_name: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("BLE device disconnected");
        ble_receiver.disconnect().await.ok();
        
        // Calculate exponential backoff delay
        let attempts_capped = (*reconnect_attempts).min(4); // Cap at 4 for 2^4 = 16 multiplier
        let delay_seconds = std::cmp::min(
            RECONNECT_DELAY_BASE * (1u64 << attempts_capped),
            RECONNECT_DELAY_MAX
        );
        
        *reconnect_attempts += 1;
        eprintln!("Attempting to reconnect to BLE device in {} seconds (attempt {})...", delay_seconds, reconnect_attempts);
        
        // Wait for backoff delay
        tokio::time::sleep(tokio::time::Duration::from_secs(delay_seconds)).await;
        
        // Attempt to reconnect with preferred device name
        match ble_receiver.connect(preferred_device_name).await {
            Ok(()) => {
                // Device name is already printed by ble_receiver.connect() via eprintln!
                println!("Reconnected! Waiting for button press to start recording...");
                *reconnect_attempts = 0; // Reset on successful reconnection
                Ok(())
            }
            Err(e) => {
                eprintln!("Failed to reconnect to BLE device: {}", e);
                Err(e.into())
            }
        }
    }
    
    // Outer loop for reconnection handling
    loop {
        // Check if input source changed
        {
            let source = input_source.lock().unwrap();
            if *source != "ble" {
                println!("Input source changed to {}, exiting BLE mode", source);
                break;
            }
        }
        
        // Get the notification stream
        let mut notifications = match ble_receiver.notifications().await {
            Ok(stream) => stream,
            Err(e) => {
                eprintln!("Failed to get notification stream: {}", e);
                // Attempt reconnection and continue outer loop
                if attempt_reconnect(&mut ble_receiver, &mut reconnect_attempts, preferred_device_name.as_deref()).await.is_ok() {
                    continue; // Successfully reconnected, get new stream
                } else {
                    continue; // Reconnection failed, retry with longer backoff
                }
            }
        };
        
        // Inner loop: process notifications with connection monitoring
        let mut connection_check_interval = tokio::time::interval(tokio::time::Duration::from_secs(3));
        connection_check_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        
        loop {
            // Check if input source changed
            {
                let source = input_source.lock().unwrap();
                if *source != "ble" {
                    println!("Input source changed to {}, exiting BLE mode", source);
                    return Ok(());
                }
            }
            
            // Use select! to monitor both notifications and connection health
            tokio::select! {
                // Check connection health periodically
                _ = connection_check_interval.tick() => {
                    if !ble_receiver.check_connection_health().await {
                        eprintln!("BLE device disconnected (connection health check failed)");
                        break; // Break inner loop to attempt reconnection
                    }
                }
                
                // Receive notifications (audio or control events) with timeout
                result = timeout(tokio::time::Duration::from_millis(100), notifications.next()) => {
                    match result {
                        Ok(Some(notification)) => {
                            // Process the notification
                            match ble_receiver.process_notification(notification) {
                                NotificationResult::Control(0x01) => {
                                    // RESP_SPEECH_START - Button pressed, start recording
                                    if !is_recording_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                                        continue; // Already recording
                                    }
                                    println!("🎤 Recording... (button pressed)");
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
                                        println!("⏹️  Stopped ({} samples, {:.2}s)", samples.len(), samples.len() as f32 / 16000.0);
                                        
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
                                        let processed_text = strip_periods_from_short_phrases(&text);
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
                                    }
                                }
                                Err(e) => {
                                    eprintln!("❌ Error: {}", e);
                                }
                            }
                        });
                                    } else {
                                        println!("⏹️  Stopped (no audio captured)");
                                    }
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
                                    
                                    // Decode Opus bundle to PCM
                                    match decoder.decode_bundle(bundle_data) {
                                        Ok(pcm_samples) => {
                                            if !pcm_samples.is_empty() {
                                                // Add to buffer while recording
                                                audio_buffer_clone.lock().unwrap().extend_from_slice(&pcm_samples);
                                                
                                                // Calculate and send audio levels for waveform visualization (BLE with reduced sensitivity)
                                                if is_recording_clone.load(Ordering::Acquire) {
                                                    let levels = calculate_audio_levels_ble(&pcm_samples);
                                                    let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                                    // Throttle to ~20fps (50ms intervals)
                                                    if last_sent.elapsed().as_millis() >= 50 {
                                                        let json = json!(levels).to_string();
                                                        println!("AUDIO_LEVELS:{}", json);
                                                        *last_sent = Instant::now();
                                                    }
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Opus decode error: {}", e);
                                        }
                                    }
                                }
                                NotificationResult::Control(_) => {
                                    // Other control events, ignore (already handled above)
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
        
        // Reconnection logic: stream ended or failed to get stream
        eprintln!("BLE device disconnected");
        
        // Clean up existing connection
        ble_receiver.disconnect().await.ok();
        
        // Calculate exponential backoff delay
        let attempts_capped = reconnect_attempts.min(4); // Cap at 4 for 2^4 = 16 multiplier
        let delay_seconds = std::cmp::min(
            RECONNECT_DELAY_BASE * (1u64 << attempts_capped),
            RECONNECT_DELAY_MAX
        );
        
        reconnect_attempts += 1;
        eprintln!("Attempting to reconnect to BLE device in {} seconds (attempt {})...", delay_seconds, reconnect_attempts);
        
        // Wait for backoff delay
        tokio::time::sleep(tokio::time::Duration::from_secs(delay_seconds)).await;
        
        // Attempt to reconnect with preferred device name
        match ble_receiver.connect(preferred_device_name.as_deref()).await {
            Ok(()) => {
                // Device name is already printed by ble_receiver.connect() via eprintln!
                println!("Reconnected! Waiting for button press to start recording...");
                reconnect_attempts = 0; // Reset on successful reconnection
            }
            Err(e) => {
                eprintln!("Failed to reconnect to BLE device: {}", e);
                // Continue outer loop to retry with longer backoff
            }
        }
    }
    
    // Clean up on exit
    ble_receiver.disconnect().await.ok();
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check INPUT_SOURCE environment variable
    let input_source = std::env::var("INPUT_SOURCE").unwrap_or_else(|_| "system".to_string());
    
    // Parse command line arguments for hotkey
    let args: Vec<String> = std::env::args().collect();
    let mut trigger_key = DEFAULT_TRIGGER_KEY;
    
    for i in 0..args.len() {
        if args[i] == "--hotkey" && i + 1 < args.len() {
            if let Some(key) = parse_hotkey(&args[i + 1]) {
                trigger_key = key;
                println!("Using hotkey: {:?}", trigger_key);
            } else {
                eprintln!("Warning: Unknown hotkey '{}', using default (Function)", args[i + 1]);
            }
            break;
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
            return rt.block_on(run_ble_audio_mode(engine_arc));
        }
        #[cfg(not(feature = "binary"))]
        {
            eprintln!("BLE mode requires binary feature");
            return Err("BLE mode not available".into());
        }
    }
    
    // Continue with system mic mode (existing code)
    // System mic typically uses 48kHz, so initialize engine with 48kHz
    println!("Loading Whisper model (48kHz for system mic)...");
    let engine = SttEngine::new_default(48000)?;
    
    println!("Warming up GPU...");
    engine.warmup()?;
    println!("Ready!");
    
    let engine = Arc::new(Mutex::new(engine));
    let audio_buffer = Arc::new(Mutex::new(Vec::<i16>::new()));
    let is_recording = Arc::new(AtomicBool::new(false));
    let is_locked = Arc::new(AtomicBool::new(false));
    let recording_stream: Arc<Mutex<Option<cpal::Stream>>> = Arc::new(Mutex::new(None));
    // Track performance history: (audio_duration_sec, realtime_factor)
    let performance_history: Arc<Mutex<VecDeque<(f32, f32)>>> = Arc::new(Mutex::new(VecDeque::with_capacity(10)));
    // Press Enter after paste flag
    let press_enter_after_paste = Arc::new(AtomicBool::new(false));
    
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("No input device found")?;
    println!("Using: {}", device.name()?);
    
    let config = device.default_input_config()?;
    let sample_rate = config.sample_rate().0;
    println!("Sample rate: {} Hz, Format: {:?}", sample_rate, config.sample_format());
    
    let audio_buffer_clone = audio_buffer.clone();
    let is_recording_clone = is_recording.clone();
    let is_locked_clone = is_locked.clone();
    let recording_stream_clone = recording_stream.clone();
    let device_clone = device.clone();
    let config_clone = config.clone();
    let engine_clone = engine.clone();
    let performance_history_clone = performance_history.clone();
    
    let (tx, rx) = mpsc::channel::<KeyEvent>();
    
    // Track key states for detecting combinations
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
                // Trigger key (configurable via --hotkey)
                EventType::KeyPress(key) if key == trigger_key_for_listener => {
                    trigger_pressed_clone.store(true, Ordering::Release);
                    
                    // Check if Control is also pressed for lock toggle
                    if control_pressed_clone.load(Ordering::Acquire) {
                        if !lock_toggle_processed_clone.swap(true, Ordering::Acquire) {
                            let _ = tx_keyboard.send(KeyEvent::ToggleLock);
                        }
                    } else {
                        // Normal recording start
                        let _ = tx_keyboard.send(KeyEvent::StartRecording);
                    }
                }
                EventType::KeyRelease(key) if key == trigger_key_for_listener => {
                    trigger_pressed_clone.store(false, Ordering::Release);
                    lock_toggle_processed_clone.store(false, Ordering::Release);
                    
                    // Only stop recording if not locked
                    if !is_locked_listener.load(Ordering::Acquire) {
                        let _ = tx_keyboard.send(KeyEvent::StopRecording);
                    }
                }
                EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => {
                    control_pressed_clone.store(true, Ordering::Release);
                    
                    // Check if Function key is also pressed for lock toggle
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
    
    // Vocabulary storage for voice commands
    #[derive(Clone)]
    struct Vocabulary {
        apps: Vec<String>,
        commands: Vec<String>,
    }
    
    let vocabulary = Arc::new(Mutex::new(Vocabulary {
        apps: Vec::new(),
        commands: Vec::new(),
    }));
    
    // Helper function to build combined prompt with app context and vocabulary
    let build_prompt = |app_name: String, window_title: String, vocab: &Vocabulary| -> Option<String> {
        let mut parts = Vec::new();
        
        // Add app context
        if !app_name.is_empty() && app_name != "Unknown" {
            if !window_title.is_empty() {
                parts.push(format!("You are transcribing for {}. The current window is: {}.", app_name, window_title));
            } else {
                parts.push(format!("You are transcribing for {}.", app_name));
            }
        }
        
        // Add vocabulary for voice commands
        if !vocab.apps.is_empty() || !vocab.commands.is_empty() {
            let mut vocab_parts = Vec::new();
            if !vocab.apps.is_empty() {
                vocab_parts.push(format!("Voice commands: open {}.", vocab.apps.join(", ")));
            }
            if !vocab.commands.is_empty() {
                vocab_parts.push(format!("Commands: {}.", vocab.commands.join(", ")));
            }
            if !vocab_parts.is_empty() {
                parts.push(vocab_parts.join(" "));
            }
        }
        
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    };
    
    // Spawn thread to read commands from stdin (for ENTER:<0|1>, INPUT_SOURCE:<system|ble>, and VOCAB: commands)
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
                        // Note: Actual switching would require restarting the process
                        // This is logged for debugging/monitoring
                    }
                } else if let Some(value) = cmd.strip_prefix("VOCAB:") {
                    // Parse vocabulary JSON: {"apps": [...], "commands": [...]}
                    if let Ok(vocab_json) = serde_json::from_str::<serde_json::Value>(value.trim()) {
                        let mut vocab = vocabulary_clone.lock().unwrap();
                        vocab.apps = vocab_json.get("apps")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        vocab.commands = vocab_json.get("commands")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                        eprintln!("MIC: Vocabulary updated: {} apps, {} commands", vocab.apps.len(), vocab.commands.len());
                    } else {
                        eprintln!("MIC: Failed to parse VOCAB command");
                    }
                }
            }
        }
    });
    
    println!("\nTrigger: Function key (or BLE device button)");
    println!("Press and hold to record, release to transcribe.");
    println!("Lock: Function+Control to toggle lock (keeps recording on)\n");
    
    // Also connect to BLE device for button press triggers (trigger-only mode)
    // This allows using BLE device as a remote trigger while audio comes from system mic
    #[cfg(feature = "binary")]
    {
        let tx_ble = tx.clone();
        let is_locked_ble = is_locked.clone();
        
        std::thread::spawn(move || {
            use ble::BleAudioReceiver;
            use futures::StreamExt;
            use tokio::time::timeout;
            
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("Failed to create tokio runtime for BLE trigger: {}", e);
                    return;
                }
            };
            
            rt.block_on(async {
                let mut ble_receiver = match BleAudioReceiver::new().await {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("Failed to initialize BLE receiver for trigger: {}", e);
                        return;
                    }
                };
                
                // Get preferred device name from environment variable
                let preferred_device_name = std::env::var("MEMO_DEVICE_NAME")
                    .ok()
                    .filter(|s| !s.is_empty() && s.to_lowercase().starts_with("memo_"));
                
                // Connect in trigger-only mode (only Control TX, not Audio Data)
                if let Err(e) = ble_receiver.connect_trigger_only(preferred_device_name.as_deref()).await {
                    eprintln!("Failed to connect to BLE device for trigger (will use keyboard only): {}", e);
                    return;
                }
                
                println!("✅ BLE device connected as trigger (audio from system mic)");
                
                // Get notification stream for button presses
                let mut notifications = match ble_receiver.notifications().await {
                    Ok(stream) => stream,
                    Err(e) => {
                        eprintln!("Failed to get BLE notification stream: {}", e);
                        return;
                    }
                };
                
                use ble::NotificationResult;
                
                loop {
                    match timeout(tokio::time::Duration::from_millis(100), notifications.next()).await {
                        Ok(Some(notification)) => {
                            match ble_receiver.process_notification(notification) {
                                NotificationResult::Control(0x01) => {
                                    // RESP_SPEECH_START - Button pressed, start recording
                                    // Just send the event - let the main loop handle state management
                                    println!("🎤 Recording... (BLE button pressed)");
                                    let _ = tx_ble.send(KeyEvent::StartRecording);
                                }
                                NotificationResult::Control(0x02) => {
                                    // RESP_SPEECH_END - Button pressed again, stop recording
                                    // Only stop if not locked - let the main loop handle state management
                                    if !is_locked_ble.load(Ordering::Acquire) {
                                        println!("⏹️  Stopped (BLE button pressed)");
                                        let _ = tx_ble.send(KeyEvent::StopRecording);
                                    }
                                }
                                _ => {
                                    // Ignore other notifications
                                }
                            }
                        }
                        Ok(None) => {
                            // Stream ended
                            eprintln!("BLE notification stream ended");
                            break;
                        }
                        Err(_) => {
                            // Timeout - continue loop
                        }
                    }
                }
            });
        });
    }
    
    loop {
        match rx.recv() {
            Ok(KeyEvent::StartRecording) => {
                if is_recording_clone.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    println!("🎤 Recording...");
                    audio_buffer_clone.lock().unwrap().clear();
                    
                    let buffer = audio_buffer_clone.clone();
                    let is_recording_for_audio = is_recording_clone.clone();
                    let last_audio_level_sent = Arc::new(Mutex::new(Instant::now()));
                    let last_audio_level_sent_clone = last_audio_level_sent.clone();
                    let stream_config = config_clone.clone().into();
                    let stream_result = match config_clone.sample_format() {
                        cpal::SampleFormat::I16 => {
                            device_clone.build_input_stream(
                                &stream_config,
                                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                    buffer.lock().unwrap().extend_from_slice(data);
                                    
                                    // Calculate and send audio levels for waveform visualization
                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = calculate_audio_levels(data);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        // Throttle to ~20fps (50ms intervals)
                                        if last_sent.elapsed().as_millis() >= 50 {
                                            let json = json!(levels).to_string();
                                            println!("AUDIO_LEVELS:{}", json);
                                            *last_sent = Instant::now();
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
                                    let mut i16_samples = Vec::with_capacity(data.len());
                                    for &s in data {
                                        let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                                        buf.push(i16_sample);
                                        i16_samples.push(i16_sample);
                                    }
                                    
                                    // Calculate and send audio levels for waveform visualization
                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = calculate_audio_levels(&i16_samples);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        // Throttle to ~20fps (50ms intervals)
                                        if last_sent.elapsed().as_millis() >= 50 {
                                            let json = json!(levels).to_string();
                                            println!("AUDIO_LEVELS:{}", json);
                                            *last_sent = Instant::now();
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
                                    let mut i16_samples = Vec::with_capacity(data.len());
                                    for &s in data {
                                        let i16_sample = ((s as i32) - 32768) as i16;
                                        buf.push(i16_sample);
                                        i16_samples.push(i16_sample);
                                    }
                                    
                                    // Calculate and send audio levels for waveform visualization
                                    if is_recording_for_audio.load(Ordering::Acquire) {
                                        let levels = calculate_audio_levels(&i16_samples);
                                        let mut last_sent = last_audio_level_sent_clone.lock().unwrap();
                                        // Throttle to ~20fps (50ms intervals)
                                        if last_sent.elapsed().as_millis() >= 50 {
                                            let json = json!(levels).to_string();
                                            println!("AUDIO_LEVELS:{}", json);
                                            *last_sent = Instant::now();
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
            Ok(KeyEvent::StopRecording) => {
                if is_recording_clone.compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
                    recording_stream_clone.lock().unwrap().take();
                    
                    let samples = {
                        let mut buf = audio_buffer_clone.lock().unwrap();
                        std::mem::take(&mut *buf)
                    };
                    
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
                        let vocabulary_for_thread = vocabulary.clone();
                        let sample_count = samples.len();
                        let audio_duration = sample_count as f32 / sample_rate as f32;
                        let start_time = Instant::now();
                        std::thread::spawn(move || {
                            println!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                            println!("🔄 Transcribing...");
                            let mut eng = engine_for_thread.lock().unwrap();
                            
                            // Capture application context and vocabulary before transcribing
                            let (app_name, window_title) = app_detection::get_application_context();
                            let vocab = vocabulary_for_thread.lock().unwrap();
                            let prompt = build_prompt(app_name, window_title, &vocab);
                            // Always set the prompt (even if empty, to clear previous context)
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
                                        let processed_text = strip_periods_from_short_phrases(&text);
                                        
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
                            println!("🎤 Recording...");
                            audio_buffer_clone.lock().unwrap().clear();
                            
                            let buffer = audio_buffer_clone.clone();
                            let is_recording_for_audio_lock = is_recording_clone.clone();
                            let last_audio_level_sent_lock = Arc::new(Mutex::new(Instant::now()));
                            let last_audio_level_sent_lock_clone = last_audio_level_sent_lock.clone();
                            let stream_config = config_clone.clone().into();
                            let stream_result = match config_clone.sample_format() {
                                cpal::SampleFormat::I16 => {
                                    device_clone.build_input_stream(
                                        &stream_config,
                                        move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                            buffer.lock().unwrap().extend_from_slice(data);
                                            
                                            // Calculate and send audio levels for waveform visualization
                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = calculate_audio_levels(data);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if last_sent.elapsed().as_millis() >= 50 {
                                                    let json = json!(levels).to_string();
                                                    println!("AUDIO_LEVELS:{}", json);
                                                    *last_sent = Instant::now();
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
                                            let mut i16_samples = Vec::with_capacity(data.len());
                                            for &s in data {
                                                let i16_sample = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
                                                buf.push(i16_sample);
                                                i16_samples.push(i16_sample);
                                            }
                                            
                                            // Calculate and send audio levels for waveform visualization
                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = calculate_audio_levels(&i16_samples);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if last_sent.elapsed().as_millis() >= 50 {
                                                    let json = json!(levels).to_string();
                                                    println!("AUDIO_LEVELS:{}", json);
                                                    *last_sent = Instant::now();
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
                                            let mut i16_samples = Vec::with_capacity(data.len());
                                            for &s in data {
                                                let i16_sample = ((s as i32) - 32768) as i16;
                                                buf.push(i16_sample);
                                                i16_samples.push(i16_sample);
                                            }
                                            
                                            // Calculate and send audio levels for waveform visualization
                                            if is_recording_for_audio_lock.load(Ordering::Acquire) {
                                                let levels = calculate_audio_levels(&i16_samples);
                                                let mut last_sent = last_audio_level_sent_lock_clone.lock().unwrap();
                                                if last_sent.elapsed().as_millis() >= 50 {
                                                    let json = json!(levels).to_string();
                                                    println!("AUDIO_LEVELS:{}", json);
                                                    *last_sent = Instant::now();
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
                            recording_stream_clone.lock().unwrap().take();
                            
                            let samples = {
                                let mut buf = audio_buffer_clone.lock().unwrap();
                                std::mem::take(&mut *buf)
                            };
                            
                            if !samples.is_empty() {
                                // Spawn transcription thread immediately for fastest response
                                let engine_for_thread = engine_clone.clone();
                                let perf_history = performance_history_clone.clone();
                                let press_enter_clone = press_enter_after_paste.clone();
                                let vocabulary_for_thread = vocabulary.clone();
                                let sample_count = samples.len();
                                let audio_duration = sample_count as f32 / sample_rate as f32;
                                let start_time = Instant::now();
                                std::thread::spawn(move || {
                                    println!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                                    println!("🔄 Transcribing...");
                                    let mut eng = engine_for_thread.lock().unwrap();
                                    
                                    // Capture application context and vocabulary before transcribing
                                    let (app_name, window_title) = app_detection::get_application_context();
                                    let vocab = vocabulary_for_thread.lock().unwrap();
                                    let prompt = build_prompt(app_name, window_title, &vocab);
                                    // Always set the prompt (even if empty, to clear previous context)
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
                                                let processed_text = strip_periods_from_short_phrases(&text);
                                                
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

