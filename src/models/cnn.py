import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ResNet-18 classifier.
def make_resnet18(num_classes: int, pretrained: bool = True):
    w = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# EfficientNet-B0 classifier.
def make_efficientnet_b0(num_classes: int, pretrained: bool = True):
    w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    m = efficientnet_b0(weights=w)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m


# Wrap a timm backbone with a classifier head.
class _TimmFeatureClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, concat_cls_patch: bool = False):
        super().__init__()
        self.backbone = backbone
        self.concat_cls_patch = concat_cls_patch
        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise ValueError("Backbone missing num_features; cannot build classifier head.")
        head_dim = feat_dim * 2 if concat_cls_patch else feat_dim
        self.classifier = nn.Linear(head_dim, num_classes)

    def _postprocess_feats(self, feats: torch.Tensor):
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.ndim == 3:
            cls_tok = feats[:, 0]
            if self.concat_cls_patch:
                if feats.shape[1] > 1:
                    patch_mean = feats[:, 1:].mean(dim=1)
                else:
                    patch_mean = cls_tok
                feats = torch.cat([cls_tok, patch_mean], dim=-1)
            else:
                feats = cls_tok
        elif self.concat_cls_patch:
            feats = torch.cat([feats, feats], dim=-1)
        return feats

    def forward(self, x):
        return self.forward_masked(x, token_keep_mask=None)

    def forward_masked(self, x, token_keep_mask=None):
        if token_keep_mask is None:
            feats = self.backbone(x)
        else:
            from src.models.vit_masking import masked_attention
            with masked_attention(self.backbone, token_keep_mask):
                feats = self.backbone(x)
        feats = self._postprocess_feats(feats)
        return self.classifier(feats)

    def extract_tokens(self, x, token_keep_mask=None):
        def _forward_features():
            if hasattr(self.backbone, "forward_features"):
                return self.backbone.forward_features(x)
            return self.backbone(x)

        if token_keep_mask is None:
            feats = _forward_features()
        else:
            from src.models.vit_masking import masked_attention
            with masked_attention(self.backbone, token_keep_mask):
                feats = _forward_features()
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        if feats.ndim == 2 and hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
        return feats


# Build a UNI-based classifier.
def make_uni(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    try:
        import timm
        from huggingface_hub import login
        from timm.layers import SwiGLUPacked
    except ImportError as exc:
        raise ImportError("UNI requires timm. Install via `pip install timm`.") from exc

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    backbone = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h",
        pretrained=pretrained,
        # UNI2-h checkpoint expects a 24-layer, 1536-dim ViT with SwiGLU + 8 reg tokens.
        patch_size=14,
        embed_dim=1536,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        mlp_ratio=2.66667 * 2,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        no_embed_class=True,
        dynamic_img_size=True,
        num_classes=0,
    )
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    return _TimmFeatureClassifier(backbone, num_classes, concat_cls_patch=False)


# Build a Virchow-based classifier.
def make_virchow(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    try:
        import timm
        from huggingface_hub import login
        from timm.layers import SwiGLUPacked
    except ImportError as exc:
        raise ImportError("Virchow requires timm. Install via `pip install timm`.") from exc

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        login(token=hf_token)

    backbone = timm.create_model(
        "hf-hub:paige-ai/Virchow",
        pretrained=pretrained,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        num_classes=0,
    )
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
    return _TimmFeatureClassifier(backbone, num_classes, concat_cls_patch=True)
