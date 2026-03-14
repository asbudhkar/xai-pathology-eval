import torch
import torch.nn.functional as F

# Store activations and gradients for CNN Grad-CAM.
class _GradCAMHook:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.act = None     # forward activations
        self.grad = None    # dscore/dact

        self._h_fwd = module.register_forward_hook(self._forward_hook)
        self._h_bwd = module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out: [B, C, H, W]
        self.act = out

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: [B, C, H, W]
        self.grad = grad_output[0]

    def close(self):
        self._h_fwd.remove()
        self._h_bwd.remove()


# Store token activations and gradients for ViT Grad-CAM.
class _TokenHook:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.act = None
        self._h_fwd = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        if isinstance(out, (list, tuple)):
            out = out[0]
        if not torch.is_tensor(out):
            raise RuntimeError("Backbone output is not a tensor; cannot compute token CAM.")
        out.retain_grad()
        self.act = out

    def close(self):
        self._h_fwd.remove()


# Return the default Grad-CAM target layer.
def get_default_target_layer(model: torch.nn.Module):
    """
    Pick a reasonable default target layer for supported CNNs.
    """
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        # EfficientNet-style models expose conv blocks in .features
        return model.features[-1]
    if hasattr(model, "backbone"):
        # ViT-style backbones output tokens, not spatial maps.
        raise ValueError("Grad-CAM requires token-based CAM for ViT backbones.")
    raise ValueError("Unsupported model for Grad-CAM; pass a target_layer explicitly.")


# Return the token layer used for ViT Grad-CAM.
def _get_vit_token_layer(backbone: torch.nn.Module):
    if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
        return backbone.blocks[-1]
    if hasattr(backbone, "layers") and len(backbone.layers) > 0:
        return backbone.layers[-1]
    raise ValueError("ViT backbone has no blocks/layers to hook for token CAM.")


# Estimate how many prefix tokens a ViT uses.
def _num_prefix_tokens(backbone: torch.nn.Module, tokens: torch.Tensor) -> int:
    n_total = int(tokens.shape[1])
    prefix = int(getattr(backbone, "num_prefix_tokens", 1))
    prefix = max(0, min(prefix, n_total))
    n_patches = n_total - prefix
    if n_patches <= 0:
        return 0
    if int(n_patches ** 0.5) ** 2 == n_patches:
        return prefix
    if n_total > 1 and int((n_total - 1) ** 0.5) ** 2 == (n_total - 1):
        return 1
    if int(n_total ** 0.5) ** 2 == n_total:
        return 0
    return prefix


# Convert token scores back to an image grid.
def _tokens_to_map(tokens: torch.Tensor, cam_tokens: torch.Tensor, backbone: torch.nn.Module, x: torch.Tensor):
    if tokens.dim() != 3:
        raise RuntimeError(f"Expected token tensor [B,N,C], got {tokens.shape}")
    prefix = _num_prefix_tokens(backbone, tokens)
    n_patches = int(tokens.shape[1] - prefix)
    gh = gw = int(n_patches ** 0.5)
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "grid_size"):
        grid = backbone.patch_embed.grid_size
        if grid[0] * grid[1] == n_patches:
            gh, gw = int(grid[0]), int(grid[1])

    cam = cam_tokens[:, prefix:]  # [1, N_patches]
    cam = cam.view(1, 1, gh, gw)
    cam = F.relu(cam)
    if cam.shape[-2:] != x.shape[-2:]:
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return cam


# Compute token-based Grad-CAM for a ViT.
def _gradcam_tokens(model: torch.nn.Module, x: torch.Tensor, target: int):
    token_layer = _get_vit_token_layer(model.backbone)
    hook = _TokenHook(token_layer)
    try:
        x = x.requires_grad_(True)
        logits = model(x)
        score = logits[0, target]
        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        tokens = hook.act
        grad = tokens.grad
        if tokens is None or grad is None:
            raise RuntimeError("Token CAM failed: missing activations or gradients.")
        if tokens.dim() != 3:
            raise RuntimeError(f"Expected token tensor [B,N,C], got {tokens.shape}")
        weights = grad.mean(dim=1, keepdim=True)  # [B,1,C]
        cam_tokens = (weights * tokens).sum(dim=2)  # [1,N]
        return _tokens_to_map(tokens, cam_tokens, model.backbone, x)
    finally:
        hook.close()


# Compute a Grad-CAM heatmap.
def gradcam_attribution(
    model: torch.nn.Module,
    x: torch.Tensor,          # [1, C, H, W]
    target: int,
    target_layer: torch.nn.Module = None,
    upsample: bool = True,
):
    """
    Returns Grad-CAM heatmap: [1, 1, H, W] aligned to input size if upsample=True.
    """
    model.eval()
    device = x.device

    if target_layer is None:
        if hasattr(model, "backbone") and not hasattr(model, "layer4") and not hasattr(model, "features"):
            return _gradcam_tokens(model, x, target)
        target_layer = get_default_target_layer(model)

    hook = _GradCAMHook(target_layer)
    try:
        x = x.requires_grad_(True)

        logits = model(x)                    # [1, num_classes]
        score = logits[0, target]

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        act = hook.act                       # [1, C, h, w]
        grad = hook.grad                     # [1, C, h, w]
        if act is None or grad is None:
            raise RuntimeError("Grad-CAM failed: missing activations or gradients. Check hooks/target_layer.")

        # weights: global-average-pooled gradients over spatial dims
        w = grad.mean(dim=(2, 3), keepdim=True)         # [1, C, 1, 1]
        cam = (w * act).sum(dim=1, keepdim=True)        # [1, 1, h, w]
        cam = F.relu(cam)

        if upsample and cam.shape[-2:] != x.shape[-2:]:
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)

        return cam.detach()

    finally:
        hook.close()

