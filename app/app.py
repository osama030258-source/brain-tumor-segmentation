import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt

IMG_SIZE = 128

# ---------------- UNet architecture (must match training) ----------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, 4, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)

# ---------------- Load model (cached so it only loads once) ----------------
@st.cache_resource
def load_model():
    model = UNet(in_channels=4, out_channels=4)
    state_dict = torch.load("models/best_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------- Preprocessing ----------------
def preprocess(image):
    image_proc = image.astype(np.float32).copy()
    for c in range(image_proc.shape[-1]):
        ch = image_proc[:, :, c]
        mx = ch.max()
        if mx > 0:
            image_proc[:, :, c] = ch / mx
    image_proc = cv2.resize(image_proc, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image_proc).permute(2, 0, 1).unsqueeze(0)
    return image_proc, tensor

def mask_to_label(mask):
    mask_resized = cv2.resize(mask.astype(np.float32), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
    label[mask_resized[:, :, 0] > 0.5] = 1
    label[mask_resized[:, :, 1] > 0.5] = 2
    label[mask_resized[:, :, 2] > 0.5] = 3
    return label

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")
st.title("🧠 Brain Tumor Segmentation from MRI Scans")
st.write("Upload a BraTS `.h5` slice file (containing `image` and `mask` datasets) to see the model's tumor segmentation.")

uploaded_file = st.file_uploader("Upload an .h5 MRI slice file", type=["h5"])

if uploaded_file is not None:
    with h5py.File(uploaded_file, "r") as f:
        image = f["image"][:]
        has_mask = "mask" in f.keys()
        mask = f["mask"][:] if has_mask else None

    model = load_model()
    image_proc, tensor = preprocess(image)

    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).numpy()

    st.subheader("Results")
    cols = st.columns(3 if has_mask else 2)

    with cols[0]:
        st.markdown("**FLAIR scan**")
        fig, ax = plt.subplots()
        ax.imshow(image_proc[:, :, 3], cmap="gray")
        ax.axis("off")
        st.pyplot(fig)

    col_idx = 1
    if has_mask:
        gt_label = mask_to_label(mask)
        with cols[1]:
            st.markdown("**Ground Truth**")
            fig, ax = plt.subplots()
            ax.imshow(gt_label, cmap="jet", vmin=0, vmax=3)
            ax.axis("off")
            st.pyplot(fig)
        col_idx = 2

    with cols[col_idx]:
        st.markdown("**Model Prediction**")
        fig, ax = plt.subplots()
        ax.imshow(pred_mask, cmap="jet", vmin=0, vmax=3)
        ax.axis("off")
        st.pyplot(fig)

    detected_classes = np.unique(pred_mask)
    class_names = {0: "Background", 1: "Necrotic core", 2: "Edema", 3: "Enhancing tumor"}
    found = [class_names[c] for c in detected_classes if c != 0]

    if found:
        st.success(f"Tumor regions detected: {', '.join(found)}")
    else:
        st.info("No tumor detected in this slice.")
else:
    st.info("👆 Upload a `.h5` file to get started.")