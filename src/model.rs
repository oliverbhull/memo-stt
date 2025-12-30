//! Model management and automatic downloading

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use crate::Result;

/// Default model to use (small.en Q5_1 - best balance)
const DEFAULT_MODEL_NAME: &str = "ggml-small.en-q5_1.bin";
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

/// Get the default model path in the user's cache directory
pub fn default_model_path() -> PathBuf {
    let cache_dir = dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|h| h.join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    
    cache_dir.join("memo-stt").join("models").join(DEFAULT_MODEL_NAME)
}

/// Ensure the model exists, downloading it if necessary
pub fn ensure_model(model_path: impl AsRef<Path>) -> Result<PathBuf> {
    let model_path = model_path.as_ref();
    
    // If model already exists, return it
    if model_path.exists() {
        return Ok(model_path.to_path_buf());
    }
    
    // If it's a relative path, try to find it in common locations
    if !model_path.is_absolute() {
        // Try current directory
        if Path::new(model_path).exists() {
            return Ok(model_path.to_path_buf());
        }
        
        // Try models/ subdirectory
        let local_path = Path::new("models").join(model_path);
        if local_path.exists() {
            return Ok(local_path);
        }
    }
    
    // Model doesn't exist - check if it's the default model name
    let model_name = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    
    if model_name == DEFAULT_MODEL_NAME || model_name.is_empty() {
        // Download default model
        let default_path = default_model_path();
        return download_model_if_needed(&default_path, DEFAULT_MODEL_NAME);
    }
    
    Err(crate::Error(format!(
        "Model not found: {}. Please download it from https://huggingface.co/ggerganov/whisper.cpp or use the default model.",
        model_path.display()
    )))
}

/// Download model if it doesn't exist
fn download_model_if_needed(dest: &Path, model_name: &str) -> Result<PathBuf> {
    // Check if already downloaded
    if dest.exists() {
        return Ok(dest.to_path_buf());
    }
    
    // Create parent directory
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| crate::Error(format!("Failed to create model directory: {}", e)))?;
    }
    
    let url = format!("{}/{}", MODEL_BASE_URL, model_name);
    
    eprintln!("📥 Downloading Whisper model (this is a one-time setup)...");
    eprintln!("   Model: {}", model_name);
    eprintln!("   URL: {}", url);
    eprintln!("   Destination: {}", dest.display());
    
    download_file(&url, dest)?;
    
    eprintln!("✅ Model downloaded successfully!");
    
    Ok(dest.to_path_buf())
}

/// Download a file from URL to destination
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(std::time::Duration::from_secs(30))
        .timeout_read(std::time::Duration::from_secs(300)) // 5 minutes for large files
        .build();
    
    let response = agent
        .get(url)
        .call()
        .map_err(|e| crate::Error(format!("Failed to download model: {}", e)))?;
    
    let total_size = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    
    let mut file = fs::File::create(dest)
        .map_err(|e| crate::Error(format!("Failed to create model file: {}", e)))?;
    
    let mut reader = response.into_reader();
    let mut buffer = [0; 8192];
    let mut downloaded = 0u64;
    let mut last_progress = 0u64;
    
    loop {
        let bytes_read = reader
            .read(&mut buffer)
            .map_err(|e| crate::Error(format!("Failed to read download: {}", e)))?;
        
        if bytes_read == 0 {
            break;
        }
        
        file.write_all(&buffer[..bytes_read])
            .map_err(|e| crate::Error(format!("Failed to write model file: {}", e)))?;
        
        downloaded += bytes_read as u64;
        
        // Print progress every 10MB
        if total_size > 0 && downloaded - last_progress > 10 * 1024 * 1024 {
            let percent = (downloaded * 100) / total_size;
            eprint!("\r   Progress: {}% ({:.1} MB / {:.1} MB)", 
                percent,
                downloaded as f64 / (1024.0 * 1024.0),
                total_size as f64 / (1024.0 * 1024.0));
            last_progress = downloaded;
        }
    }
    
    if total_size > 0 && downloaded != total_size {
        return Err(crate::Error(format!(
            "Incomplete download: expected {} bytes, got {}",
            total_size, downloaded
        )));
    }
    
    eprintln!(); // New line after progress
    
    Ok(())
}




