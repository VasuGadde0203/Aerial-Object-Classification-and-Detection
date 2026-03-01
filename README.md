# 🛩️ Aerial Object Classification & Detection
### Bird vs Drone — Deep Learning with PyTorch & YOLOv8

<br>

> A computer vision system that classifies and detects aerial objects in real-time, with applications in airport safety, wildlife monitoring, and restricted airspace surveillance.

<br>

---

## 📌 Project Overview

This project builds an end-to-end **computer vision pipeline** capable of:

- 🐦 Classifying aerial images as **Bird** or **Drone** using a fine-tuned ResNet18
- 🎯 Detecting and localizing birds and drones using **YOLOv8** object detection
- 🌐 Deploying the classification model through an interactive **Streamlit** web app

### Real-World Applications

| Domain | Use Case |
|--------|----------|
| ✈️ Aviation Safety | Airport bird-strike prevention & drone intrusion alerts |
| 🦅 Wildlife Protection | Monitoring bird populations near wind farms or flight paths |
| 🔒 Airspace Security | Detecting unauthorized drones in restricted zones |
| 🔬 Environmental Research | Automated aerial fauna tracking |

---

## 🧠 Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Deep Learning | PyTorch, Torchvision |
| Object Detection | YOLOv8 (Ultralytics) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Model Persistence | `.pth` (PyTorch checkpoint) |

---

## 📂 Project Structure

```
Aerial-Object-Classification/
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Dataset exploration & visualization
│   ├── 02_Custom_CNN.ipynb           # Custom CNN training
│   └── 03_ResNet18_Transfer.ipynb    # Transfer learning training
│
├── models/
│   ├── custom_cnn.pth                # Saved Custom CNN weights
│   └── bird_drone_resnet18.pth       # Saved ResNet18 weights (final model)
│
├── data/
│   ├── train/
│   │   ├── bird/
│   │   └── drone/
│   ├── valid/
│   │   ├── bird/
│   │   └── drone/
│   └── test/
│       ├── bird/
│       └── drone/
│
├── app.py                            # Streamlit deployment app
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 📊 Dataset Summary

### Image Specification
- **Resolution:** 416 × 416 pixels
- **Format:** JPG / PNG
- **Classes:** Bird, Drone (binary classification)

### Dataset Distribution

| Split | Bird | Drone | Total |
|-------|------|-------|-------|
| Train | 1,414 | 1,248 | 2,662 |
| Valid | 217 | 225 | 442 |
| Test | 121 | 94 | 215 |
| **Total** | **1,752** | **1,567** | **3,319** |

The dataset is **reasonably balanced** across both classes, minimizing class-bias risk during training.

### Data Augmentation (Training Set)
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast)
- Normalization (ImageNet mean/std)

---

## 🧠 Model Development

### 1️⃣ Custom CNN (Baseline)

A custom convolutional neural network built from scratch as the baseline model.

**Architecture:**
```
Input (3 × 416 × 416)
    → Conv2D(32) + BatchNorm + ReLU + MaxPool
    → Conv2D(64) + BatchNorm + ReLU + MaxPool
    → Conv2D(128) + BatchNorm + ReLU + MaxPool
    → Conv2D(256) + BatchNorm + ReLU + MaxPool
    → Flatten
    → FC(512) + Dropout(0.5)
    → FC(2) → Softmax
```

| Metric | Score |
|--------|-------|
| Validation F1 Score | ~0.87 |
| Test Accuracy | ~87% |
| Convergence | Moderate (30–40 epochs) |

---

### 2️⃣ Transfer Learning — ResNet18 ✅ Final Model

Fine-tuned **ResNet18** pretrained on ImageNet for binary aerial object classification.

**Strategy:**
- Loaded pretrained ResNet18 weights (ImageNet)
- Froze backbone layers to retain learned low-level features
- Replaced final fully connected layer: `FC(512 → 2)`
- Trained only the classifier head initially
- Applied learning rate scheduling for stable convergence

| Metric | Score |
|--------|-------|
| Validation F1 Score | **~0.95** |
| Test Accuracy | **95%** |
| Convergence | Fast (15–20 epochs) |

### Test Set Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Bird | 0.94 | 0.98 | **0.96** | 121 |
| Drone | 0.97 | 0.93 | **0.95** | 94 |
| **Weighted Avg** | **0.96** | **0.95** | **0.95** | **215** |

The model generalizes well with **minimal overfitting** across both classes.

---

## 🔍 Model Comparison

| Model | Test Accuracy | F1 Score | Parameters | Convergence |
|-------|--------------|----------|------------|-------------|
| Custom CNN | ~87% | ~0.87 | ~2.1M | Moderate |
| **ResNet18** | **95%** | **0.95** | ~11.2M | **Fast** |

**Winner: ResNet18** — ImageNet pretrained features transfer effectively to aerial imagery, delivering 8% accuracy gain with faster training.

---

## 🎯 Object Detection — YOLOv8

A separate detection pipeline built using YOLOv8 for **localizing** birds and drones within images using bounding boxes.

### Model Configuration

| Setting | Value |
|---------|-------|
| Architecture | YOLOv8n (Nano) |
| Framework | Ultralytics |
| Input Size | 416 × 416 |
| Classes | Bird, Drone |
| Training Images | ~2,700 |
| Label Format | YOLO (.txt — normalized xywh) |

### Detection Dataset Structure
```
detection_data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Training Command
```bash
yolo task=detect mode=train \
     model=yolov8n.pt \
     data=data.yaml \
     epochs=50 \
     imgsz=416
```

---

## 🚀 Streamlit Deployment

An interactive web interface for real-time classification.

### Features
- 📤 Upload any aerial image (JPG / PNG)
- 🤖 Instant Bird / Drone prediction using ResNet18
- 📊 Confidence score display
- 🎨 Clean, minimal UI

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Aerial-Object-Classification.git
cd Aerial-Object-Classification

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
streamlit>=1.32.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=10.0.0
```

---

## 📈 Key Observations

- **Transfer learning significantly outperforms** the custom CNN — pretrained ImageNet features (edges, textures, shapes) transfer well to aerial object imagery.
- **ResNet18 converges faster** (≈15 epochs vs ≈35 epochs) due to pretrained weight initialization.
- **Slight drone recall drop (0.93)** compared to bird recall (0.98) suggests:
  - Drones appear in more varied poses and lighting conditions
  - Dataset has slightly fewer drone samples
  - Potential fix: hard negative mining, targeted augmentation for drones

---

## ⚠️ Limitations

- Dataset size is moderate (~3,300 images) — performance may vary on highly diverse real-world imagery.
- Some **content bias** present: studio/clean-background images may not generalize perfectly to natural scenes.
- Model tested on **static images only** — not validated on real-time video streams.
- YOLOv8 trained on **nano model** — larger variants (YOLOv8s/m) would yield better detection mAP.
- Not a production-grade surveillance system — further robustness testing required.

---

## 🔮 Future Improvements

- [ ] Fine-tune ResNet deeper layers (unfreeze later blocks)
- [ ] Add **Grad-CAM** heatmap visualization to explain predictions
- [ ] Train larger YOLOv8 variant (YOLOv8s or YOLOv8m) on GPU
- [ ] Real-time **webcam detection** pipeline
- [ ] Deploy to **Streamlit Cloud** or **HuggingFace Spaces**
- [ ] Add video inference support (frame-by-frame detection)
- [ ] Experiment with EfficientNet / ViT for classification

---

## 🏁 Conclusion

This project demonstrates a complete **end-to-end deep learning workflow**:

✅ Dataset preparation and augmentation  
✅ Custom CNN architecture design  
✅ Transfer learning with ResNet18  
✅ Model evaluation and comparison  
✅ Object detection pipeline using YOLOv8  
✅ Interactive deployment via Streamlit  

The **ResNet18-based classifier** achieved **95% test accuracy** with strong generalization, while the YOLOv8 detection model successfully localizes aerial objects with bounding boxes.

This system demonstrates practical viability for bird-strike prevention, drone surveillance, and environmental monitoring applications.

---

## 👤 Author

**Vasu Gadde**  
Deep Learning & Data Science Enthusiast

---

