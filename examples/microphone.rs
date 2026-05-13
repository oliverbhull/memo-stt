//! Microphone recording is implemented in the standalone CLI rather than in
//! the library, so this example just points you at the right entry point.
//!
//! Install and run the CLI:
//! ```bash
//! cargo install memo-stt --features binary
//! memo-stt
//! ```

fn main() {
    println!("Microphone capture lives in the memo-stt CLI.");
    println!();
    println!("Install and run with:");
    println!("    cargo install memo-stt --features binary");
    println!("    memo-stt");
    println!();
    println!("For library usage with your own audio source, see examples/basic.rs.");
}
