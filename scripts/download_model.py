#!/usr/bin/env python3
"""
Download Moonshine Tiny INT8 quantized models (encoder + decoder) and tokenizer from HuggingFace.

Usage:
    python scripts/download_model.py

Requirements:
    pip install huggingface_hub
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

MODELS_DIR = Path(__file__).parent.parent / "models"
# Use UsefulSensors/moonshine (official repository)
REPO_ID = "UsefulSensors/moonshine"

# INT8 quantized model paths
ENCODER_PATH = "onnx/merged/tiny/quantized/encoder_model.onnx"
DECODER_PATH = "onnx/merged/tiny/quantized/decoder_model_merged.onnx"

def main():
    print("Downloading Moonshine Tiny INT8 quantized model files...")
    print(f"Repository: {REPO_ID}")
    print("Quantization: INT8 (recommended for size/accuracy balance)")
    print()
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check available files in the repository
        print("Checking available files...")
        files = list_repo_files(REPO_ID, repo_type="model")
        
        # Verify INT8 quantized models exist
        encoder_exists = ENCODER_PATH in files
        decoder_exists = DECODER_PATH in files
        
        if not encoder_exists or not decoder_exists:
            print("⚠️  INT8 quantized models not found. Checking alternatives...")
            # Fallback to float32 if INT8 not available
            float_encoder = "onnx/merged/tiny/float/encoder_model.onnx"
            float_decoder = "onnx/merged/tiny/float/decoder_model_merged.onnx"
            
            if float_encoder in files and float_decoder in files:
                print("Found float32 models. Using as fallback (larger size).")
                encoder_path = float_encoder
                decoder_path = float_decoder
            else:
                print("❌ Could not find required model files")
                print("Available files (first 30):")
                for file in files[:30]:
                    print(f"  - {file}")
                return 1
        else:
            encoder_path = ENCODER_PATH
            decoder_path = DECODER_PATH
        
        # Find tokenizer - prefer English tiny tokenizer
        tokenizer_file = None
        for file in files:
            if file.endswith("tokenizer.json"):
                # Prefer English tiny tokenizer
                if "tiny" in file.lower() and "/tiny-ar/" not in file and "/tiny-zh/" not in file:
                    tokenizer_file = file
                    print(f"Found tokenizer: {file}")
                    break
                elif not tokenizer_file:  # Fallback to any tokenizer
                    tokenizer_file = file
        
        if not tokenizer_file:
            # Try root level tokenizer
            tokenizer_file = "tokenizer.json"
            print(f"Trying root level tokenizer: {tokenizer_file}")
        
        # Download encoder model
        print(f"\nDownloading encoder model: {encoder_path}...")
        encoder_downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=encoder_path,
            local_dir=str(MODELS_DIR),
        )
        
        # Rename encoder to standard name
        target_encoder = MODELS_DIR / "encoder_model.onnx"
        if encoder_downloaded != str(target_encoder):
            if target_encoder.exists():
                target_encoder.unlink()
            os.rename(encoder_downloaded, str(target_encoder))
            print(f"Saved as {target_encoder.name}")
        
        # Download decoder model
        print(f"Downloading decoder model: {decoder_path}...")
        decoder_downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=decoder_path,
            local_dir=str(MODELS_DIR),
        )
        
        # Rename decoder to standard name
        target_decoder = MODELS_DIR / "decoder_model_merged.onnx"
        if decoder_downloaded != str(target_decoder):
            if target_decoder.exists():
                target_decoder.unlink()
            os.rename(decoder_downloaded, str(target_decoder))
            print(f"Saved as {target_decoder.name}")
        
        # Download tokenizer
        print(f"Downloading tokenizer: {tokenizer_file}...")
        tokenizer_path = None
        try:
            tokenizer_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=tokenizer_file,
                local_dir=str(MODELS_DIR),
            )
        except Exception as e:
            print(f"Warning: Could not download tokenizer from {tokenizer_file}: {e}")
            print("Trying alternative locations...")
            # Try common tokenizer locations
            for alt_tokenizer in ["tokenizer.json", "tokenizers/tiny/tokenizer.json"]:
                try:
                    tokenizer_path = hf_hub_download(
                        repo_id=REPO_ID,
                        filename=alt_tokenizer,
                        local_dir=str(MODELS_DIR),
                    )
                    print(f"Downloaded tokenizer from {alt_tokenizer}")
                    break
                except:
                    continue
        
        # Copy tokenizer to expected location if it's in a subdirectory
        if tokenizer_path and tokenizer_path != str(MODELS_DIR / "tokenizer.json"):
            target_tokenizer = MODELS_DIR / "tokenizer.json"
            if not target_tokenizer.exists():
                import shutil
                shutil.copy2(tokenizer_path, str(target_tokenizer))
                print(f"Copied tokenizer to {target_tokenizer}")
        
        print()
        print("✅ Model files downloaded successfully!")
        print("Files:")
        total_size = 0
        for file in MODELS_DIR.glob("*.onnx"):
            size = file.stat().st_size / (1024 * 1024)
            total_size += size
            print(f"  {file.name}: {size:.2f} MB")
        for file in MODELS_DIR.glob("*.json"):
            size = file.stat().st_size / 1024
            total_size += size / 1024
            print(f"  {file.name}: {size:.2f} KB")
        
        print(f"\nTotal model size: {total_size:.2f} MB")
        print("(Target: < 50MB including runtime binary)")
            
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        import traceback
        traceback.print_exc()
        print()
        print("Alternative: Download manually from:")
        print(f"  https://huggingface.co/{REPO_ID}")
        print()
        print("Required files:")
        print("  - onnx/merged/tiny/quantized/encoder_model.onnx")
        print("  - onnx/merged/tiny/quantized/decoder_model_merged.onnx")
        print("  - tokenizer.json")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

