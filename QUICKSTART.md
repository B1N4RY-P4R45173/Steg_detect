# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Project Structure
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Prepare Dataset

**Option A: Use Sample Data (for testing)**
```bash
# Download a small sample
python download_dataset.py --kaggle
```

**Option B: Full BOSSbase Dataset**
- Download from: https://dde.binghamton.edu/download/
- Extract to `data/cover/`

### 4. Generate Stego Images
```bash
python create_stego_images.py --input data/cover/ --output data/stego/ --algorithm RANDOM
```

### 5. Train the Model
```bash
jupyter notebook steganography_detection.ipynb
```

Then run all cells (Cell → Run All)

---

## 📊 Expected Results

After training (with 5000+ images):
- **Accuracy**: 92-96%
- **Training time**: 30-60 minutes (GPU) or 2-4 hours (CPU)
- **Model size**: ~50 MB

---

## 🔧 Common Commands

### Analyze Dataset
```bash
python preprocess_data.py analyze --directory data/cover/
```

### Resize Images
```bash
python preprocess_data.py resize --input data/cover/ --output data/cover_resized/ --width 256 --height 256
```

### Convert to Grayscale
```bash
python preprocess_data.py grayscale --input data/cover/ --output data/cover_gray/
```

---

## 💡 Tips

1. **GPU Acceleration**: Ensure CUDA is installed for faster training
2. **Memory Issues**: Reduce batch size in config (line ~35 in notebook)
3. **Small Dataset**: Works with as few as 1000 images (500 cover + 500 stego)
4. **Quick Test**: Use `MAX_IMAGES = 500` in config for rapid prototyping

---

## 📚 Next Steps

- Read full documentation in `README.md`
- Experiment with different architectures
- Try various steganography algorithms
- Deploy model for inference

---

**Questions?** Check the troubleshooting section in README.md
