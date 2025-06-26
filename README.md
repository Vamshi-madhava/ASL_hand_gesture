#  ASL Alphabet Image Classifier

A high-performance deep learning model trained on American Sign Language (ASL) alphabet images using TensorFlow and Google Colab. This project demonstrates data preprocessing, mixed precision training, and GPU-accelerated model training using the `tf.data` API.

---

##  Dataset

- **Dataset Source:** [Kaggle – ASL Alphabet]
- **Structure:** 87,000 images across 29 classes (A–Z, `nothing`, `space`, `delete`)
- **Image Resolution Used:** `100x100` (configurable)

---

## Features

-   Mixed precision (`float16`) training with GPU acceleration
-   Fast data pipeline using `tf.data`
-   Early stopping and model checkpointing
-   Model saved in modern `.keras` format
-   Ready for Colab with Drive integration

---

##  Setup (Colab)

> No local installation needed. This project is built for Google Colab.

1. Download the dataset zip from Kaggle.
2. Upload `archive.zip` to your **Google Drive > MyDrive**.
3. Open the Colab notebook and run all cells.

---

##  Model Architecture

```text
Input: 100x100x3 RGB image
├── Conv2D (32 filters, 3x3) + ReLU
├── MaxPooling2D
├── Conv2D (64 filters, 3x3) + ReLU
├── MaxPooling2D
├── Flatten
├── Dense (128) + ReLU
├── Dropout (0.4)
└── Dense (29) + Softmax
