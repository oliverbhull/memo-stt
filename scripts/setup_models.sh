#!/bin/bash
# Setup script following Moonshine official instructions
# This creates a Python venv and downloads the models for the Rust project

set -e

echo "Setting up Moonshine models for memo-stt..."
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Create virtual environment (following Moonshine instructions)
VENV_DIR="venv_moonshine"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install huggingface_hub for downloading models
echo "Installing huggingface_hub..."
pip install --quiet huggingface_hub

# Create models directory
MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

# Download models using our Python script
echo ""
echo "Downloading Moonshine Tiny ONNX model and tokenizer..."
python3 scripts/download_model.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Models are now in the $MODELS_DIR/ directory."
echo "You can now run: cargo run --example trigger_mic"
echo ""
echo "To deactivate the virtual environment, run: deactivate"

