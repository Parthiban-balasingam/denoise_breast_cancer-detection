from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

try:
    import timm  # optional only for old checkpoints; Xception/DarkNet53 are skipped in this version
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = APP_DIR / "streamlit_models"
IMAGE_SIZE = 224


def load_image_any_depth(img: Image.Image) -> Image.Image:
    if img.mode in ["I;16", "I", "F"]:
        arr = np.array(img).astype(np.float32)
        arr -= arr.min()
        maxv = arr.max() if arr.max() > 0 else 1.0
        arr = (arr / maxv * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def make_transform(image_size: int = IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MReMarNetApprox(nn.Module):
    def __init__(self, num_classes: int, base_width: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes
        layers = []
        in_ch = 3
        for i in range(10):
            layers.append(ConvBlock(in_ch, base_width))
            in_ch = base_width
            if i in {1, 3, 5, 7}:
                layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(base_width, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.prototypes = nn.Parameter(torch.randn(num_classes, base_width))
        nn.init.xavier_uniform_(self.prototypes)
        self.relation = nn.Sequential(
            nn.Linear(base_width * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        fmap = self.encoder(x)
        feat = self.pool(fmap).flatten(1)
        feat = self.dropout(feat)
        fc_hidden = F.relu(self.fc1(feat))
        fc_logits = self.fc2(fc_hidden)
        feat_norm = F.normalize(feat, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        rel_scores = []
        for c in range(self.num_classes):
            proto = proto_norm[c].unsqueeze(0).expand_as(feat_norm)
            rel_in = torch.cat([feat_norm, proto, torch.abs(feat_norm - proto)], dim=1)
            rel_scores.append(self.relation(rel_in))
        relation_scores = torch.cat(rel_scores, dim=1)
        fc_prob = torch.softmax(fc_logits, dim=1)
        rel_prob = torch.softmax(relation_scores, dim=1)
        final_prob = 0.5 * fc_prob + 0.5 * rel_prob
        final_logits = torch.log(final_prob + 1e-8)
        return {"logits": final_logits}


class ViProtoDenoiseNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        dropout: float = 0.4,
        use_denoising: bool = True,
        use_mask_aware: bool = True,
        use_modality_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.use_denoising = use_denoising
        self.use_mask_aware = use_mask_aware
        self.use_modality_bias = use_modality_bias
        base = models.efficientnet_b0(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        feat_dim = base.classifier[1].in_features
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)
        gate_in_dim = embed_dim * 2 + 3
        self.noise_gate = nn.Sequential(
            nn.Linear(gate_in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.modality_embed = nn.Embedding(2, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _masked_pool(self, fmap: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            pooled = self.avgpool(fmap).flatten(1)
            lesion_ratio = torch.zeros((fmap.size(0), 1), device=fmap.device)
            return pooled, lesion_ratio
        mask_small = F.interpolate(mask.float(), size=fmap.shape[-2:], mode="bilinear", align_corners=False)
        lesion_mass = mask_small.sum(dim=(2, 3), keepdim=True)
        lesion_ratio = mask_small.mean(dim=(2, 3)).clamp(0, 1)
        global_feat = self.avgpool(fmap).flatten(1)
        lesion_feat = (fmap * mask_small).sum(dim=(2, 3)) / (lesion_mass.squeeze(-1).squeeze(-1) + 1e-6)
        pooled = torch.where((lesion_ratio > 0).expand_as(global_feat), 0.7 * global_feat + 0.3 * lesion_feat, global_feat)
        return pooled, lesion_ratio

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, modality_id: Optional[torch.Tensor] = None, group_id: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        fmap = self.features(x)
        pooled, lesion_ratio = self._masked_pool(fmap, mask if self.use_mask_aware else None)
        emb = self.embedding(pooled)
        if modality_id is None:
            modality_id = torch.zeros((x.size(0),), dtype=torch.long, device=x.device)
        if self.use_modality_bias:
            emb = emb + 0.05 * self.modality_embed(modality_id)
        emb_norm = F.normalize(emb, p=2, dim=1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=1)
        similarity = emb_norm @ proto_norm.t()
        relation_prob = torch.softmax(similarity, dim=1)
        uncertainty = 1.0 - relation_prob.max(dim=1, keepdim=True).values
        ref_idx = similarity.argmax(dim=1)
        ref_proto = self.prototypes[ref_idx]
        modality_scalar = modality_id.float().unsqueeze(1)
        gate_inputs = torch.cat([emb, ref_proto, uncertainty, lesion_ratio, modality_scalar], dim=1)
        gate = self.noise_gate(gate_inputs)
        denoised = gate * emb + (1.0 - gate) * ref_proto if self.use_denoising else emb
        logits = self.classifier(denoised)
        return {"logits": logits, "uncertainty": uncertainty, "similarity": similarity}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "AlexNet":
        model = models.alexnet(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "GoogLeNet":
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "MReMarNet":
        return MReMarNetApprox(num_classes=num_classes)
    if model_name == "ViProtoDenoiseNet":
        return ViProtoDenoiseNet(num_classes=num_classes, use_denoising=True, use_mask_aware=True, use_modality_bias=True)
    if model_name == "ViProto_no_denoise":
        return ViProtoDenoiseNet(num_classes=num_classes, use_denoising=False, use_mask_aware=True, use_modality_bias=True)
    if model_name == "ViProto_no_adaptive_margin":
        return ViProtoDenoiseNet(num_classes=num_classes, use_denoising=True, use_mask_aware=True, use_modality_bias=True)
    if model_name == "ViProto_no_modality_constraints":
        return ViProtoDenoiseNet(num_classes=num_classes, use_denoising=True, use_mask_aware=False, use_modality_bias=False)
    if model_name in {"Xception", "DarkNet53"}:
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is not installed. Install it with: pip install timm")
        timm_name = "xception" if model_name == "Xception" else "darknet53"
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


SKIP_MODEL_NAMES = {"Xception", "DarkNet53"}


def find_model_infos(output_root: Path):
    infos = []
    for p in output_root.rglob("*_streamlit_info.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("model_name") in SKIP_MODEL_NAMES:
                continue
            checkpoint = Path(data.get("checkpoint_path", ""))
            if not checkpoint.exists():
                checkpoint = p.parent / f"{data['model_name']}_fold{data.get('selected_fold', 1)}_best.pth"
            if checkpoint.exists():
                data["checkpoint_path"] = str(checkpoint)
                data["info_path"] = str(p)
                infos.append(data)
        except Exception:
            pass
    return sorted(infos, key=lambda x: (x.get("dataset_name", ""), x.get("task_name", ""), x.get("model_name", "")))


@st.cache_resource(show_spinner=True)
def load_trained_model(info_json: str):
    info = json.loads(info_json)
    class_names = info["class_names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(info["model_name"], len(class_names)).to(device)
    state = torch.load(info["checkpoint_path"], map_location=device)
    try:
        model.load_state_dict(state, strict=True)
        load_note = "Loaded with strict=True"
    except Exception as e:
        missing, unexpected = model.load_state_dict(state, strict=False)
        load_note = f"Loaded with strict=False. Missing={len(missing)}, unexpected={len(unexpected)}. Original error: {e}"
    model.eval()
    return model, device, load_note


def predict_image(model: nn.Module, device: torch.device, image: Image.Image, image_size: int, model_name: str):
    tfm = make_transform(image_size)
    image = load_image_any_depth(image)
    x = tfm(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if model_name.startswith("ViProto"):
            mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
            modality_id = torch.zeros((1,), dtype=torch.long, device=device)
            output = model(x, mask=mask, modality_id=modality_id)
        else:
            output = model(x)
        logits = output["logits"] if isinstance(output, dict) else output
        probs = torch.softmax(logits.float(), dim=1).detach().cpu().numpy()[0]
    return probs


st.set_page_config(page_title="Breast Cancer Single-Fold Classifier", layout="wide")
st.title("Breast Cancer Image Classification — Single-Fold Model")
st.caption("Standalone Streamlit app. It loads checkpoints generated by the Jupyter notebook.")

with st.sidebar:
    st.header("Model folder")
    output_root_text = st.text_input("Output root", value=str(DEFAULT_OUTPUT_ROOT))
    output_root = Path(output_root_text)
    st.write("Device:", "CUDA/GPU" if torch.cuda.is_available() else "CPU")

infos = find_model_infos(output_root)
if not infos:
    st.error("No trained model metadata found. Finish the notebook training first, then rerun this app.")
    st.stop()

labels = [f"{i['dataset_name']} / {i['task_name']} / {i['model_name']} / fold {i.get('selected_fold', 1)}" for i in infos]
choice = st.sidebar.selectbox("Choose trained checkpoint", labels)
info = infos[labels.index(choice)]
info_json = json.dumps(info, sort_keys=True)

model, device, load_note = load_trained_model(info_json)
st.sidebar.success(load_note)
st.sidebar.write("Checkpoint:")
st.sidebar.code(info["checkpoint_path"])

class_names = info["class_names"]
image_size = int(info.get("image_size", IMAGE_SIZE))

uploaded = st.file_uploader("Upload one breast ultrasound/mammogram image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("Upload an image to get prediction probabilities.")
    st.stop()

image = Image.open(uploaded)
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

probs = predict_image(model, device, image, image_size, info["model_name"])
pred_idx = int(np.argmax(probs))
pred_class = class_names[pred_idx]
confidence = float(probs[pred_idx])

with col2:
    st.subheader("Prediction")
    st.metric("Predicted class", pred_class)
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    prob_df = pd.DataFrame({"Class": class_names, "Probability": probs})
    st.dataframe(prob_df, use_container_width=True, hide_index=True)
    st.bar_chart(prob_df.set_index("Class"))

st.warning("Research-use output only. Do not use this app for clinical diagnosis.")
