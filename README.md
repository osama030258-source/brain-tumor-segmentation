# Brain Tumor Segmentation from MRI Scans

An end-to-end deep learning system that segments brain tumors from multi-modal MRI scans using a U-Net architecture, trained on the BraTS2020 dataset.

## Overview

This project automates brain tumor segmentation from MRI scans to assist in medical diagnosis and treatment planning. It detects three tumor sub-regions — **necrotic core**, **edema**, and **enhancing tumor** — from four MRI modalities (T1, T1c, T2, FLAIR).

## Dataset

- **Source:** [BraTS2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) (Kaggle)
- **Format:** Preprocessed `.h5` slices, each containing:
  - `image`: shape `(240, 240, 4)` — 4 stacked MRI modalities
  - `mask`: shape `(240, 240, 3)` — 3 tumor sub-region masks
- **Size:** 57,198 total slices; trained on an 18,000-slice subset (15,300 train / 2,700 validation)

## Model

- **Architecture:** U-Net (encoder-decoder with skip connections), implemented in PyTorch
- **Input:** 4-channel MRI slice, resized to 128×128
- **Output:** 4-class pixel-wise segmentation (background, necrotic, edema, enhancing)
- **Training:** 10 epochs, Adam optimizer, CrossEntropyLoss, trained on Kaggle's free T4 GPU

## Results

| Metric | Score |
|---|---|
| Dice Coefficient | 0.755 |
| IoU | 0.628 |
| Pixel Accuracy | 0.997* |

\* *Pixel accuracy is inflated by class imbalance (most pixels are background). Dice and IoU are the more meaningful metrics here, since they measure performance on the tumor regions specifically.*

## Project Structure

brain_tumor_segmentation/

├── app/

│   └── app.py              # Streamlit web app for inference

├── data/                    # Sample .h5 MRI slices for testing

├── models/

│   └── best_model.pth       # Trained U-Net weights

├── notebooks/

│   └── brain_tumor_segmentation.ipynb   # Local inference pipeline + validation

├── requirements.txt

└── README.md
## How It Works

1. **Preprocessing:** Each MRI modality channel is normalized (divided by its max value) and resized to 128×128.
2. **Inference:** The preprocessed 4-channel tensor is passed through the trained U-Net.
3. **Output:** A pixel-wise class map is produced, then color-coded by tumor sub-region for visualization.

## Running the App

```bash
git clone https://github.com/osama030258-source/brain-tumor-segmentation.git
cd brain-tumor-segmentation
pip install -r requirements.txt
streamlit run app/app.py
```

Upload a BraTS `.h5` slice file in the browser to see the segmentation result.

## Tools & Technologies

- **Language:** Python
- **Deep Learning:** PyTorch
- **Image Processing:** OpenCV, h5py
- **Visualization:** Matplotlib
- **Web App:** Streamlit
- **Training Environment:** Kaggle Notebooks (GPU T4 x2)

## Limitations & Honest Notes

- Trained on an 18,000-slice subset (not the full 57,198) due to time constraints — performance would likely improve with full-dataset training and more epochs.
- Pixel accuracy is not a reliable standalone metric here due to background-pixel dominance; Dice/IoU are reported as the primary metrics.
- The model operates on individual 2D slices, not full 3D volumes.