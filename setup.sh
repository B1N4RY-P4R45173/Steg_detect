#!/bin/bash

# Setup script for Steganography Detection Project

echo "=========================================="
echo "Steganography Detection - Setup Script"
echo "=========================================="
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/cover
mkdir -p data/stego
mkdir -p models
mkdir -p results
mkdir -p scripts

echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt
echo "✓ Requirements installed"
echo ""

# Check TensorFlow installation
echo "Verifying TensorFlow installation..."
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
echo ""

# Print next steps
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download BOSSbase dataset to data/cover/"
echo "2. Generate stego images using: python scripts/create_stego_images.py"
echo "3. Run Jupyter notebook: jupyter notebook steganography_detection.ipynb"
echo ""
echo "For more information, see README.md"
echo ""
