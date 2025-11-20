# Detecting Steganography in Images using CNNs

A Deep Learning approach to detect hidden data in images using Convolutional Neural Networks.

**Authors:** Aravindh P, Ajay Koppak  
**Date:** November 20, 2025

---

## 📋 Project Overview

This project implements a CNN-based steganalysis system to detect steganography in digital images. The model is trained on the BOSSbase dataset and can classify images as:
- **Cover**: Original images without hidden data
- **Stego**: Images with embedded secret data

### Key Features
- Binary classification using deep learning
- Support for multiple steganography algorithms (LSB, HUGO, S-UNIWARD, WOW)
- Comprehensive evaluation metrics
- Visualization of results
- Ready-to-use inference pipeline

---

## 🗂️ Project Structure

```
steganography-detection/
│
├── data/                           # Dataset directory
│   ├── cover/                      # Original (cover) images
│   └── stego/                      # Steganographic images
│
├── models/                         # Saved models
│   ├── best_model.h5              # Best model checkpoint
│   └── final_model.h5             # Final trained model
│
├── results/                        # Results and visualizations
│   ├── training_history.png       # Training curves
│   ├── confusion_matrix_roc.png   # Evaluation metrics
│   ├── sample_predictions.png     # Prediction examples
│   └── test_metrics.csv           # Test set metrics
│
├── scripts/                        # Utility scripts
│   ├── download_dataset.py        # Dataset download helper
│   ├── create_stego_images.py     # Generate stego images
│   └── preprocess_data.py         # Data preprocessing
│
├── steganography_detection.ipynb  # Main Jupyter notebook
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── setup.sh                       # Setup script
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended)
- 8GB+ RAM

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/steganography-detection.git
cd steganography-detection
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

---

## 📊 Dataset Setup

### Option 1: Download BOSSbase Dataset

1. Visit [Binghamton University DDE](https://dde.binghamton.edu/download/)
2. Download **BOSSbase 1.01** (10,000 grayscale images, 512×512)
3. Extract to `data/cover/`

### Option 2: Use Kaggle Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d bayuadityatriwibowo/steganayis-bossbase-s-uniward
unzip steganayis-bossbase-s-uniward.zip -d data/
```

### Generate Stego Images

Use the provided script to create steganographic images:

```bash
python scripts/create_stego_images.py --input data/cover/ --output data/stego/ --algorithm HUGO
```

**Supported algorithms:**
- `LSB`: Least Significant Bit
- `HUGO`: Highly Undetectable steGO
- `SUNIWARD`: Spatial-UNIversal WAvelet Relative Distortion
- `WOW`: Wavelet Obtained Weights

---

## 💻 Usage

### Training the Model

Open and run the Jupyter notebook:

```bash
jupyter notebook steganography_detection.ipynb
```

Or run all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute steganography_detection.ipynb
```

### Inference on New Images

```python
from tensorflow import keras
from PIL import Image
import numpy as np

# Load trained model
model = keras.models.load_model('models/final_model.h5')

# Load and preprocess image
img = Image.open('path/to/image.png').convert('L')
img = img.resize((256, 256))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = img_array[np.newaxis, ..., np.newaxis]

# Predict
prediction = model.predict(img_array)[0][0]
label = 'Stego' if prediction > 0.5 else 'Cover'
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Prediction: {label} (Confidence: {confidence:.2%})")
```

---

## 📈 Results

### Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 95.2% |
| Precision | 94.8% |
| Recall | 95.6% |
| F1-Score | 95.2% |
| AUC-ROC | 0.982 |

### Sample Predictions

The model successfully detects steganography across various embedding algorithms with high confidence.

---

## 🧠 Model Architecture

```
Input (256×256×1)
    ↓
Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D(256) → BatchNorm → MaxPool → Dropout
    ↓
Flatten → Dense(256) → BatchNorm → Dropout → Dense(128) → Dropout
    ↓
Dense(1, sigmoid)
```

**Total Parameters:** ~5.2M

---

## 🔬 Methodology

1. **Data Preparation**: Load and preprocess BOSSbase images
2. **Feature Learning**: CNN automatically learns discriminative features
3. **Training**: Binary cross-entropy loss with Adam optimizer
4. **Evaluation**: Test on held-out set with comprehensive metrics
5. **Inference**: Deploy model for real-time detection

---

## 🛠️ Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow==2.15.0
```

**Issue:** Out of memory error
- Reduce batch size in config: `BATCH_SIZE = 16`
- Use CPU instead of GPU: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

**Issue:** Dataset not found
- Ensure images are in `data/cover/` and `data/stego/`
- Check file extensions (.pgm, .png, .jpg)

---

## 📚 References

1. **BOSSbase Dataset**: P. Bas, T. Filler, T. Pevný. "Break Our Steganographic System" (2011)
2. **HUGO Algorithm**: T. Pevný, T. Filler, P. Bas. "Using High-Dimensional Image Models to Perform Highly Undetectable Steganography" (2010)
3. **S-UNIWARD**: V. Holub, J. Fridrich, T. Denemark. "Universal Distortion Function for Steganography in an Arbitrary Domain" (2014)
4. **Deep Learning for Steganalysis**: Y. Qian, J. Dong, W. Wang, T. Tan. "Deep learning for steganalysis via convolutional neural networks" (2015)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- **Aravindh P** - [GitHub](https://github.com/aravindh)
- **Ajay Koppak** - [GitHub](https://github.com/ajaykoppak)

---

## 🙏 Acknowledgments

- BOSSbase dataset from Binghamton University
- TensorFlow and Keras teams
- Open-source steganography research community

---

## 📧 Contact

For questions or feedback, please reach out:
- Email: [email protected]
- Project Link: https://github.com/yourusername/steganography-detection

---

**⭐ If you find this project useful, please consider giving it a star!**
