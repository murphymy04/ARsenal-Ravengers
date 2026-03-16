"""Light-ASD model architecture (CVPR 2023).

Reproduced from https://github.com/Junhua-Liao/Light-ASD (MIT License).

Inputs:
  audioFeature : (B, T_audio, 13)  — MFCC features, T_audio = 4 * T_visual
  visualFeature: (B, T_visual, 112, 112) — grayscale face crops uint8

ASDInference.predict() returns per-frame speaking probability in [0, 1].
"""

import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _AudioBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0), bias=False)
        self.bn_m_3 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.t_3 = nn.Conv2d(out_ch, out_ch, (1, 3), padding=(0, 1), bias=False)
        self.bn_t_3 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.m_5 = nn.Conv2d(in_ch, out_ch, (5, 1), padding=(2, 0), bias=False)
        self.bn_m_5 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.t_5 = nn.Conv2d(out_ch, out_ch, (1, 5), padding=(0, 2), bias=False)
        self.bn_t_5 = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.last = nn.Conv2d(out_ch, out_ch, (1, 1), bias=False)
        self.bn_last = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x3 = self.relu(self.bn_m_3(self.m_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_m_5(self.m_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


class _VisualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_down=False):
        super().__init__()
        self.relu = nn.ReLU()
        s = (1, 2, 2) if is_down else (1, 1, 1)
        self.s_3 = nn.Conv3d(in_ch, out_ch, (1, 3, 3), stride=s, padding=(0, 1, 1), bias=False)
        self.bn_s_3 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.t_3 = nn.Conv3d(out_ch, out_ch, (3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn_t_3 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.s_5 = nn.Conv3d(in_ch, out_ch, (1, 5, 5), stride=s, padding=(0, 2, 2), bias=False)
        self.bn_s_5 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.t_5 = nn.Conv3d(out_ch, out_ch, (5, 1, 1), padding=(2, 0, 0), bias=False)
        self.bn_t_5 = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)
        self.last = nn.Conv3d(out_ch, out_ch, (1, 1, 1), bias=False)
        self.bn_last = nn.BatchNorm3d(out_ch, momentum=0.01, eps=1e-3)

    def forward(self, x):
        x3 = self.relu(self.bn_s_3(self.s_3(x)))
        x3 = self.relu(self.bn_t_3(self.t_3(x3)))
        x5 = self.relu(self.bn_s_5(self.s_5(x)))
        x5 = self.relu(self.bn_t_5(self.t_5(x5)))
        return self.relu(self.bn_last(self.last(x3 + x5)))


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class _VisualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _VisualBlock(1, 32, is_down=True)
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        # x: (B, 1, T, 112, 112)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = x.transpose(1, 2)                    # (B, T, C, W, H)
        B, T, C, W, H = x.shape
        x = x.reshape(B * T, C, W, H)
        x = self.maxpool(x)                       # (B*T, C, 1, 1)
        return x.view(B, T, C)                    # (B, T, 128)


class _AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
        self.block3 = _AudioBlock(64, 128)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def forward(self, x):
        # x: (B, 1, 13, T_audio)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)                        # (B, 128, 13, T_audio//4)
        x = torch.mean(x, dim=2, keepdim=True)    # (B, 128, 1, T_audio//4)
        x = x.squeeze(2).transpose(1, 2)          # (B, T_audio//4, 128)
        return x


# ---------------------------------------------------------------------------
# BGRU classifier
# ---------------------------------------------------------------------------

class _BGRU(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        # Field names must match the checkpoint keys exactly
        self.gru_forward = nn.GRU(ch, ch, batch_first=True)
        self.gru_backward = nn.GRU(ch, ch, batch_first=True)
        self.gelu = nn.GELU()
        for m in self.modules():
            if isinstance(m, nn.GRU):
                nn.init.kaiming_normal_(m.weight_ih_l0)
                nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_(); m.bias_hh_l0.data.zero_()

    def forward(self, x):
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, [1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, [1])
        return self.gelu(x)


# ---------------------------------------------------------------------------
# ASD_Model  (matches original repo naming for weight compatibility)
# ---------------------------------------------------------------------------

class ASD_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.GRU = _BGRU(128)

    def forward_visual_frontend(self, x):
        # x: (B, T, 112, 112) uint8
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255.0 - 0.4161) / 0.1688
        return self.visualEncoder(x)              # (B, T, 128)

    def forward_audio_frontend(self, x):
        # x: (B, T_audio, 13)
        x = x.unsqueeze(1).transpose(2, 3)        # (B, 1, 13, T_audio)
        return self.audioEncoder(x)               # (B, T_audio//4, 128)

    def forward_audio_visual_backend(self, x1, x2):
        # x1: audio (B, T, 128), x2: visual (B, T, 128)
        x = x1 + x2
        x = self.GRU(x)
        return x.reshape(-1, 128)                 # (B*T, 128)

    def forward_visual_backend(self, x):
        return x.reshape(-1, 128)                 # (B*T, 128)

    def forward(self, audioFeature, visualFeature):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        outsAV = self.forward_audio_visual_backend(audioEmbed, visualEmbed)
        outsV = self.forward_visual_backend(visualEmbed)
        return outsAV, outsV


# ---------------------------------------------------------------------------
# Inference wrapper — includes the lossAV FC head for scoring
# ---------------------------------------------------------------------------

class ASDInference(nn.Module):
    """Load pretrained weights and score face crops for speaking probability."""

    # GitHub LFS URLs for pretrained weights (try in order)
    _WEIGHT_URLS = [
        "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/finetuning_TalkSet.model",
        "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/pretrain_AVA_CVPR.model",
    ]

    def __init__(self):
        super().__init__()
        self.model = ASD_Model()
        # lossAV.FC from the original repo
        self.lossAV = nn.Linear(128, 2)

    @classmethod
    def load(cls, weights_path: Path, device: str = "cpu") -> "ASDInference":
        """Load pretrained weights, downloading if necessary."""
        _ensure_weights(weights_path, cls._WEIGHT_URLS)
        net = cls()
        state = torch.load(weights_path, map_location=device, weights_only=False)
        own = net.state_dict()
        loaded = 0
        for k, v in state.items():
            # Map: model.* → model.*, lossAV.FC.* → lossAV.*
            if k.startswith("lossAV.FC."):
                own_key = k.replace("lossAV.FC.", "lossAV.")
            elif k in own:
                own_key = k
            else:
                continue
            if own_key in own and own[own_key].shape == v.shape:
                own[own_key].copy_(v)
                loaded += 1
        net.load_state_dict(own)
        net.eval()
        net.to(torch.device(device))
        print(f"Light-ASD: loaded {loaded} weight tensors from {weights_path.name}")
        return net

    @torch.no_grad()
    def predict(
        self,
        visual: np.ndarray,   # (T, 112, 112) uint8 grayscale
        audio: np.ndarray,    # (T*4, 13) float32 MFCC
    ) -> float:
        """Return mean speaking probability in [0, 1] for this window."""
        dev = next(self.parameters()).device
        vt = torch.from_numpy(visual).float().unsqueeze(0).to(dev)  # (1, T, 112, 112)
        at = torch.from_numpy(audio).unsqueeze(0).float().to(dev)   # (1, T*4, 13)
        outsAV, _ = self.model(at, vt)                               # (T, 128)
        logits = self.lossAV(outsAV)                                 # (T, 2)
        probs = F.softmax(logits, dim=-1)[:, 1]                      # (T,) speaking prob
        return float(probs.mean().item())


# ---------------------------------------------------------------------------
# Weight download helper
# ---------------------------------------------------------------------------

def _ensure_weights(path: Path, urls: list[str]):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Light-ASD weights → {path.name} …")
    last_err = None
    for url in urls:
        try:
            def _prog(b, bs, total):
                if total > 0:
                    pct = min(b * bs / total * 100, 100)
                    bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                    print(f"\r  [{bar}] {pct:.1f}%", end="", flush=True)
            urllib.request.urlretrieve(url, path, reporthook=_prog)
            print(f"\n  Done.")
            return
        except Exception as e:
            last_err = e
            print(f"\n  Failed ({url}): {e}")
    raise RuntimeError(
        f"Could not download Light-ASD weights. Last error: {last_err}\n"
        "Please clone https://github.com/Junhua-Liao/Light-ASD and copy\n"
        "weight/finetuning_TalkSet.model to your ar-glasses/data/ directory."
    )
