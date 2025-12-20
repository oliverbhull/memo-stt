//! Microphone recording example
//!
//! This example shows how to record from the microphone and transcribe in real-time.
//! Run with: `cargo run --example microphone`
//!
//! Note: This requires the binary dependencies (cpal, rdev) which are only
//! available when running the main binary, not as a library example.
//!
//! For library usage, see the basic example.

fn main() {
    println!("Microphone recording example");
    println!("For full microphone recording, use the main binary:");
    println!("  cargo run --bin memo-stt");
    println!("\nFor library usage, see examples/basic.rs");
}
