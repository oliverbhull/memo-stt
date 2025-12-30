#!/bin/bash
# Download Moonshine Tiny INT8 quantized models (encoder + decoder) and tokenizer from HuggingFace

set -e

MODELS_DIR="models"
REPO_ID="UsefulSensors/moonshine"

# INT8 quantized model paths
ENCODER_PATH="onnx/merged/tiny/quantized/encoder_model.onnx"
DECODER_PATH="onnx/merged/tiny/quantized/decoder_model_merged.onnx"

echo "Downloading Moonshine Tiny INT8 quantized model files..."
echo "Repository: $REPO_ID"
echo "Quantization: INT8 (recommended for size/accuracy balance)"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing..."
    pip install huggingface_hub
fi

# Download encoder model
echo "Downloading encoder model: $ENCODER_PATH..."
huggingface-cli download "$REPO_ID" "$ENCODER_PATH" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False

# Rename encoder to standard name
if [ -f "$MODELS_DIR/$ENCODER_PATH" ]; then
    mv "$MODELS_DIR/$ENCODER_PATH" "$MODELS_DIR/encoder_model.onnx"
    echo "Saved as encoder_model.onnx"
fi

# Download decoder model
echo "Downloading decoder model: $DECODER_PATH..."
huggingface-cli download "$REPO_ID" "$DECODER_PATH" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False

# Rename decoder to standard name
if [ -f "$MODELS_DIR/$DECODER_PATH" ]; then
    mv "$MODELS_DIR/$DECODER_PATH" "$MODELS_DIR/decoder_model_merged.onnx"
    echo "Saved as decoder_model_merged.onnx"
fi

# Download tokenizer
echo "Downloading tokenizer.json..."
huggingface-cli download "$REPO_ID" "tokenizer.json" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False || \
huggingface-cli download "$REPO_ID" "tokenizers/tiny/tokenizer.json" --local-dir "$MODELS_DIR" --local-dir-use-symlinks False || \
echo "Warning: Could not find tokenizer.json in expected locations"

# Copy tokenizer to root if it's in a subdirectory
if [ -f "$MODELS_DIR/tokenizers/tiny/tokenizer.json" ] && [ ! -f "$MODELS_DIR/tokenizer.json" ]; then
    cp "$MODELS_DIR/tokenizers/tiny/tokenizer.json" "$MODELS_DIR/tokenizer.json"
    echo "Copied tokenizer to models/tokenizer.json"
fi

echo ""
echo "✅ Model files downloaded successfully!"
echo "Files:"
ls -lh "$MODELS_DIR"/*.onnx "$MODELS_DIR"/*.json 2>/dev/null || true
echo ""
echo "Total size:"
du -sh "$MODELS_DIR" 2>/dev/null || true





