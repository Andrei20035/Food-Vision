# 🍔 Food Vision — EfficientNetB0 Transfer Learning on Food101

Deep learning image classification project built with **TensorFlow** using **EfficientNetB0** and the **Food101 dataset (101 classes, 100k+ images)**.

This project demonstrates a **production-style transfer learning workflow** including mixed precision training, fine-tuning, optimized input pipelines, and training callbacks for stability and performance.

---

## 🚀 Project Highlights

- Transfer learning with **EfficientNetB0**
- **Mixed precision training** (`float16`) for faster GPU throughput
- Fine-tuning the base model after feature extraction
- Scalable **TFDS input pipeline** (batched, shuffled, preprocessed)
- **EarlyStopping** to prevent overfitting
- **ModelCheckpoint** for best-weight recovery
- Training on **Food101** (real-world, large-scale dataset)

---

## 🧠 Model Workflow

1. Load **Food101** via `tensorflow_datasets`
2. Use pretrained **EfficientNetB0** as feature extractor
3. Evaluate baseline performance
4. Unfreeze base model layers for **fine-tuning**
5. Enable **mixed precision policy**
6. Train with:
   - `Adam` optimizer  
   - `sparse_categorical_crossentropy`  
   - Early stopping  
   - Checkpointing
7. Plot training vs validation loss curves

---

## 🏗️ Tech Stack

- **TensorFlow / Keras**
- **EfficientNetB0**
- **TensorFlow Datasets (TFDS)**
- Mixed precision training
- Google Colab (GPU training)

---

## 📊 Dataset

**Food101**
- 101 food categories  
- 101,000 images  
- Real-world noisy labels  
- Standard benchmark for food classification

## ⚙️ Training Configuration

| Parameter          | Value                                  |
|--------------------|----------------------------------------|
Batch size           | 32                                     |
Loss function        | Sparse categorical crossentropy        |
Optimizer            | Adam (learning rate = 1e-4)            |
Precision policy     | Mixed float16                          |
Callbacks            | EarlyStopping, ModelCheckpoint         |
Epochs               | Up to 100 (early stopped)              |

---

## 📈 Results

- Baseline evaluation performed before fine-tuning  
- Fine-tuning improved validation performance  
- Early stopping prevented overfitting  
- Best model weights saved automatically  
- Training vs validation loss curves plotted to verify convergence  

---

## 🧩 Key ML Concepts Demonstrated

- Transfer learning vs fine-tuning  
- Mixed precision for performance optimization  
- Large-scale dataset handling with TFDS  
- Callback-driven training control  
- Reproducible training pipeline  

---

## 📌 Future Improvements

- Top-1 / Top-5 accuracy reporting  
- Confusion matrix & class-wise metrics  
- Data augmentation pipeline  
- Model export (SavedModel / TFLite)  
- Inference API (FastAPI or TensorFlow Serving)  
