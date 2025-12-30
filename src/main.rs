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
mod app_detection;

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

fn inject_text(text: &str) -> Result<(), Box<dyn std::error::Error>> {
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
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        let mut enigo = Enigo::new();
        let paste_mod = EnigoKey::Control;
        enigo.key_down(paste_mod);
        enigo.key_click(EnigoKey::Layout('v'));
        enigo.key_up(paste_mod);
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    
    println!("Loading Whisper model...");
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
    std::thread::spawn(move || {
        listen(move |event: Event| {
            match event.event_type {
                // Trigger key (configurable via --hotkey)
                EventType::KeyPress(key) if key == trigger_key_for_listener => {
                    trigger_pressed_clone.store(true, Ordering::Release);
                    
                    // Check if Control is also pressed for lock toggle
                    if control_pressed_clone.load(Ordering::Acquire) {
                        if !lock_toggle_processed_clone.swap(true, Ordering::Acquire) {
                            let _ = tx.send(KeyEvent::ToggleLock);
                        }
                    } else {
                        // Normal recording start
                        let _ = tx.send(KeyEvent::StartRecording);
                    }
                }
                EventType::KeyRelease(key) if key == trigger_key_for_listener => {
                    trigger_pressed_clone.store(false, Ordering::Release);
                    lock_toggle_processed_clone.store(false, Ordering::Release);
                    
                    // Only stop recording if not locked
                    if !is_locked_listener.load(Ordering::Acquire) {
                        let _ = tx.send(KeyEvent::StopRecording);
                    }
                }
                EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => {
                    control_pressed_clone.store(true, Ordering::Release);
                    
                    // Check if Function key is also pressed for lock toggle
                    if trigger_pressed_clone.load(Ordering::Acquire) {
                        if !lock_toggle_processed_clone.swap(true, Ordering::Acquire) {
                            let _ = tx.send(KeyEvent::ToggleLock);
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
    
    println!("\nTrigger: Function key");
    println!("Press and hold to record, release to transcribe.");
    println!("Lock: Function+Control to toggle lock (keeps recording on)\n");
    
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
                        // Spawn transcription thread immediately for fastest response
                        let engine_for_thread = engine_clone.clone();
                        let perf_history = performance_history_clone.clone();
                        let sample_count = samples.len();
                        let audio_duration = sample_count as f32 / sample_rate as f32;
                        let start_time = Instant::now();
                        std::thread::spawn(move || {
                            println!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                            println!("🔄 Transcribing...");
                            let mut eng = engine_for_thread.lock().unwrap();
                            
                            // Capture application context before transcribing
                            let (app_name, window_title) = app_detection::get_application_context();
                            let prompt = if !app_name.is_empty() && app_name != "Unknown" {
                                if !window_title.is_empty() {
                                    format!("You are transcribing for {}. The current window is: {}.", app_name, window_title)
                                } else {
                                    format!("You are transcribing for {}.", app_name)
                                }
                            } else {
                                String::new()
                            };
                            // Always set the prompt (even if empty, to clear previous context)
                            eng.set_prompt(if prompt.is_empty() { None } else { Some(prompt) });
                            
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
                                        
                                        // Output FINAL: JSON for Electron app integration
                                        let json_output = json!({
                                            "rawTranscript": text,
                                            "processedText": text,
                                            "wasProcessedByLLM": false,
                                            "appContext": {
                                                "appName": app_name,
                                                "windowTitle": window_title
                                            }
                                        });
                                        println!("FINAL: {}", json_output);
                                        
                                        // Inject first for fastest response time
                                        let inject_start = Instant::now();
                                        match inject_text(&text) {
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
                                let sample_count = samples.len();
                                let audio_duration = sample_count as f32 / sample_rate as f32;
                                let start_time = Instant::now();
                                std::thread::spawn(move || {
                                    println!("⏹️  Stopped ({} samples, {:.2}s)", sample_count, audio_duration);
                                    println!("🔄 Transcribing...");
                                    let mut eng = engine_for_thread.lock().unwrap();
                                    
                                    // Capture application context before transcribing
                                    let (app_name, window_title) = app_detection::get_application_context();
                                    let prompt = if !app_name.is_empty() && app_name != "Unknown" {
                                        if !window_title.is_empty() {
                                            format!("You are transcribing for {}. The current window is: {}.", app_name, window_title)
                                        } else {
                                            format!("You are transcribing for {}.", app_name)
                                        }
                                    } else {
                                        String::new()
                                    };
                                    // Always set the prompt (even if empty, to clear previous context)
                                    eng.set_prompt(if prompt.is_empty() { None } else { Some(prompt) });
                                    
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
                                                
                                                // Output FINAL: JSON for Electron app integration
                                                let json_output = json!({
                                                    "rawTranscript": text,
                                                    "processedText": text,
                                                    "wasProcessedByLLM": false,
                                                    "appContext": {
                                                        "appName": app_name,
                                                        "windowTitle": window_title
                                                    }
                                                });
                                                println!("FINAL: {}", json_output);
                                                
                                                // Inject first for fastest response time
                                                let inject_start = Instant::now();
                                                match inject_text(&text) {
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

