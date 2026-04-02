import os
import traceback
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Callable, List, Sequence

import torch

from .utils.io import get_weights_path, has_internet_connection


def encoder_factory(model_name: str, **kwargs) -> torch.nn.Module:
    """Instantiate a patch encoder model by name."""
    if model_name in encoder_registry:
        return encoder_registry[model_name](**kwargs)
    raise ValueError(f"Unknown encoder name {model_name}")


class BasePatchEncoder(torch.nn.Module):
    _has_internet = has_internet_connection()
    
    def __init__(self, weights_path: Optional[str] = None, **build_kwargs: Dict[str, Any]):
        super().__init__()
        self.enc_name: Optional[str] = None
        self.weights_path: Optional[str] = weights_path
        self.model, self.eval_transforms, self.precision = self._build(**build_kwargs)

    def ensure_valid_weights_path(self, weights_path: str) -> None:
        if weights_path and not os.path.exists(weights_path):
            raise FileNotFoundError(f"Expected checkpoint at '{weights_path}', but the path was not found.")
    
    def ensure_has_internet(self, enc_name: str) -> None:
        if not BasePatchEncoder._has_internet:
            raise FileNotFoundError(
                "Internet connection does seem not available. Auto checkpoint download is disabled."
                f"To proceed, please manually download: {enc_name},\n"
                "and place it in the model registry in:\n`trident/patch_encoder_models/local_ckpts.json`"
            )
        
    def _get_weights_path(self) -> str:
        if self.weights_path:
            self.ensure_valid_weights_path(self.weights_path)
            return self.weights_path

        weights_path = get_weights_path("patch", self.enc_name)
        self.ensure_valid_weights_path(weights_path)
        return weights_path

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
    @abstractmethod
    def _build(self, **build_kwargs: Dict[str, Any]) -> Tuple[torch.nn.Module, Callable, torch.dtype]:
        pass


class ViTFeatureSelectorMixin:
    """Optional ViT intermediate-feature extraction utilities."""

    _SUPPORTED_FEATURE_LAYERS = {"ln2", "fc1", "act", "rc2"}
    _SUPPORTED_FUSION = {"mean", "concat"}

    def _setup_feature_selector(
        self,
        backbone: torch.nn.Module,
        *,
        feature_layer: Optional[str] = None,
        block_index: Optional[int] = None,
        block_indices: Optional[Sequence[int]] = None,
        fusion: str = "mean",
        token_pool: str = "cls",
    ) -> None:
        self._selector_backbone = backbone
        self._selector_num_blocks = len(backbone.blocks)
        self._selector_num_prefix_tokens = getattr(backbone, "num_prefix_tokens", 1)
        self._selector_token_pool = token_pool
        self._selector_fusion = fusion.lower()
        self._selector_enabled = any(
            v is not None for v in (feature_layer, block_index, block_indices)
        )

        if not self._selector_enabled:
            return

        self._selector_feature_layer = "rc2" if feature_layer is None else feature_layer.lower()
        if self._selector_feature_layer not in self._SUPPORTED_FEATURE_LAYERS:
            raise ValueError(
                f"Unsupported feature_layer='{feature_layer}'. "
                f"Supported values: {sorted(self._SUPPORTED_FEATURE_LAYERS)}"
            )

        if self._selector_fusion not in self._SUPPORTED_FUSION:
            raise ValueError(
                f"Unsupported fusion='{fusion}'. Supported values: {sorted(self._SUPPORTED_FUSION)}"
            )

        self._selector_block_indices = self._resolve_block_indices(
            num_blocks=self._selector_num_blocks,
            block_index=block_index,
            block_indices=block_indices,
        )

    def _resolve_block_indices(
        self,
        *,
        num_blocks: int,
        block_index: Optional[int],
        block_indices: Optional[Sequence[int]],
    ) -> List[int]:
        if block_index is not None and block_indices is not None:
            raise ValueError("Only one of block_index and block_indices can be set.")

        if block_index is not None:
            block_indices = [block_index]

        if block_indices is None:
            return [num_blocks - 1]

        normalized: List[int] = []
        for idx in block_indices:
            idx = int(idx)
            if idx < 0:
                idx = num_blocks + idx
            if idx < 0 or idx >= num_blocks:
                raise ValueError(f"block index {idx} out of range [0, {num_blocks - 1}]")
            normalized.append(idx)

        seen = set()
        ordered = []
        for idx in normalized:
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return ordered
    
    def _resolve_target_module(self, block: torch.nn.Module, feature_layer: str) -> torch.nn.Module:
        if feature_layer == "rc2":
            return block

        if feature_layer == "ln2":
            if hasattr(block, "norm2"):
                return block.norm2
            raise ValueError("Target block has no `norm2`, cannot extract ln2 feature.")

        if not hasattr(block, "mlp"):
            raise ValueError("Target block has no `mlp`, cannot extract fc1/act feature.")
        mlp = block.mlp

        if feature_layer == "fc1":
            for name in ("fc1", "w12", "in_proj"):
                if hasattr(mlp, name):
                    return getattr(mlp, name)
            raise ValueError("Cannot find a supported fc1 module in block.mlp.")

        if feature_layer == "act":
            for name in ("act", "act_layer", "activation"):
                if hasattr(mlp, name):
                    return getattr(mlp, name)
            raise ValueError("Cannot find a supported activation module in block.mlp.")

        raise ValueError(f"Unknown feature_layer={feature_layer}")

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim != 3:
            raise ValueError(f"Expected 2D/3D tensor for feature pooling, got shape={tuple(x.shape)}")

        n_prefix = int(self._selector_num_prefix_tokens)
        token_pool = self._selector_token_pool

        if token_pool == "cls":
            if n_prefix > 0:
                return x[:, 0]
            return x.mean(dim=1)

        if token_pool == "mean":
            start = n_prefix if x.shape[1] > n_prefix else 0
            return x[:, start:].mean(dim=1)

        if token_pool == "cls_mean":
            cls = x[:, 0] if n_prefix > 0 else x.mean(dim=1)
            start = n_prefix if x.shape[1] > n_prefix else 0
            patch_mean = x[:, start:].mean(dim=1)
            return torch.cat([cls, patch_mean], dim=-1)

        raise ValueError("token_pool must be one of {'cls', 'mean', 'cls_mean'}")

    def _extract_selected_vit_feature(self, x: torch.Tensor) -> torch.Tensor:
        captures: Dict[int, torch.Tensor] = {}
        handles = []

        for idx in self._selector_block_indices:
            block = self._selector_backbone.blocks[idx]
            target_module = self._resolve_target_module(block, self._selector_feature_layer)

            def _save_output(_module, _input, output, block_idx=idx):
                captures[block_idx] = output[0] if isinstance(output, tuple) else output

            handles.append(target_module.register_forward_hook(_save_output))

        try:
            _ = self._selector_backbone.forward_features(x)
        finally:
            for h in handles:
                h.remove()

        missing = [idx for idx in self._selector_block_indices if idx not in captures]
        if missing:
            raise RuntimeError(
                f"Failed to capture features for block indices {missing}. "
                f"Requested layer={self._selector_feature_layer}."
            )

        features = [self._pool_tokens(captures[idx]) for idx in self._selector_block_indices]
        if len(features) == 1:
            return features[0]
        if self._selector_fusion == "mean":
            return torch.stack(features, dim=0).mean(dim=0)
        return torch.cat(features, dim=-1)


class Conchv1InferenceEncoder(ViTFeatureSelectorMixin, BasePatchEncoder):
    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(
        self, 
        with_proj: bool = False,
        normalize: bool = False,
        feature_layer: Optional[str] = None,
        block_index: Optional[int] = None,
        block_indices: Optional[Sequence[int]] = None,
        fusion: str = "mean",
        token_pool: str = "cls",
    ):
        self.enc_name = "conch"
        self.with_proj = with_proj
        self.normalize = normalize

        try:
            from .model_zoo.conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        except Exception:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")

        weights_path = self._get_weights_path()
        if weights_path:
            try:
                model, eval_transform = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=weights_path)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create CONCH v1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/CONCH."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model, eval_transform = create_model_from_pretrained(
                    "conch_ViT-B-16", checkpoint_path="hf_hub:MahmoodLab/conch"
                )
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download CONCH v1 model, make sure that you were granted access "
                    "and that you correctly registered your token"
                )

        self.tokenizer = get_tokenizer()
        precision = torch.float32

        self._setup_feature_selector(
            model.visual.trunk,
            feature_layer=feature_layer,
            block_index=block_index,
            block_indices=block_indices,
            fusion=fusion,
            token_pool=token_pool,
        )
        return model, eval_transform, precision
    
    def forward(self, x):
        if getattr(self, "_selector_enabled", False):
            return self._extract_selected_vit_feature(x)
        return self.model.encode_image(x, proj_contrast=self.with_proj, normalize=self.normalize)

    def _from_text_to_embeddings(self, texts, device: str):
        from .model_zoo.conch.open_clip_custom import tokenize

        tokenized_prompts = tokenize(texts=texts, tokenizer=self.tokenizer)
        tokenized_prompts = tokenized_prompts.to(device)
        return self.model.encode_text(tokenized_prompts)

    def run_zero_shot(self, texts, image_features: torch.Tensor, device: str):
        from torch.nn import functional as F

        image_features = image_features.to(device)
        text_features = self._from_text_to_embeddings(texts, device)
        logit_scale = self.model.logit_scale.exp()
        similarity = torch.matmul(image_features, text_features.T) * logit_scale
        probs = F.softmax(similarity, dim=-1)
        return probs.detach()


class UNIv2InferenceEncoder(ViTFeatureSelectorMixin, BasePatchEncoder):
    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(
        self, 
        feature_layer: Optional[str] = None,
        block_index: Optional[int] = None,
        block_indices: Optional[Sequence[int]] = None,
        fusion: str = "mean",
        token_pool: str = "mean",
    ):
        import timm
        from torchvision import transforms

        self.enc_name = "uni2"
        weights_path = self._get_weights_path()

        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }

        if weights_path:
            try:
                model = timm.create_model(model_name="vit_giant_patch14_224", pretrained=False, **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create UNI2-h model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/MahmoodLab/UNI2-h."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download UNI v2 model, make sure that you were granted access "
                    "and that you correctly registered your token"
                )

        eval_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

        self._setup_feature_selector(
            model,
            feature_layer=feature_layer,
            block_index=block_index,
            block_indices=block_indices,
            fusion=fusion,
            token_pool=token_pool,
        )
        return model, eval_transform, torch.bfloat16

    def forward(self, x):
        if getattr(self, "_selector_enabled", False):
            return self._extract_selected_vit_feature(x)
        return self.model(x)


class Virchow2InferenceEncoder(ViTFeatureSelectorMixin, BasePatchEncoder):
    import timm
    
    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(
        self,
        return_cls: bool = False,
        timm_kwargs={"mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU},
        feature_layer: Optional[str] = None,
        block_index: Optional[int] = None,
        block_indices: Optional[Sequence[int]] = None,
        fusion: str = "mean",
        token_pool: str = "cls_mean",
    ):
        import timm
        import torchvision
        from torchvision import transforms

        self.enc_name = "virchow2"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "img_size": 224,
                    "init_values": 1e-5,
                    "num_classes": 0,
                    "reg_tokens": 4,
                    "mlp_ratio": 5.3375,
                    "global_pool": "",
                    "dynamic_img_size": True,
                    "mlp_layer": timm.layers.SwiGLUPacked,
                    "act_layer": torch.nn.SiLU,
                }
                model = timm.create_model("vit_huge_patch14_224", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create Virchow2 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/paige-ai/Virchow2."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, **timm_kwargs)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download Virchow-2 model, make sure that you were granted access "
                    "and that you correctly registered your token"
                )
        
        eval_transform = transforms.Compose(
            [
                transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.return_cls = return_cls
        
        self._setup_feature_selector(
            model,
            feature_layer=feature_layer,
            block_index=block_index,
            block_indices=block_indices,
            fusion=fusion,
            token_pool=token_pool,
        )
        return model, eval_transform, torch.float16

    def forward(self, x):
        if getattr(self, "_selector_enabled", False):
            return self._extract_selected_vit_feature(x)
    
        output = self.model(x)
        class_token = output[:, 0]
        if self.return_cls:
            return class_token
        
        patch_tokens = output[:, 5:]
        return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


class HOptimus1InferenceEncoder(ViTFeatureSelectorMixin, BasePatchEncoder):
    def __init__(self, **build_kwargs):
        super().__init__(**build_kwargs)

    def _build(
        self,
        timm_kwargs={"init_values": 1e-5, "dynamic_img_size": False},
        feature_layer: Optional[str] = None,
        block_index: Optional[int] = None,
        block_indices: Optional[Sequence[int]] = None,
        fusion: str = "mean",
        token_pool: str = "cls",
        **kwargs,
    ):
        import timm
        from torchvision import transforms

        assert (
            timm.__version__ == "0.9.16"
        ), f"H-Optimus requires timm version 0.9.16, but found {timm.__version__}. Please install the correct version using `pip install timm==0.9.16`"

        self.enc_name = "hoptimus1"
        weights_path = self._get_weights_path()

        if weights_path:
            try:
                timm_kwargs = {
                    "num_classes": 0,
                    "img_size": 224,
                    "global_pool": "token",
                    "init_values": 1e-5,
                    "dynamic_img_size": False,
                }
                model = timm.create_model("vit_giant_patch14_reg4_dinov2", **timm_kwargs)
                model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    f"Failed to create H-Optimus-1 model from local checkpoint at '{weights_path}'. "
                    "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-1."
                )
        else:
            self.ensure_has_internet(self.enc_name)
            try:
                model = timm.create_model("hf-hub:bioptimus/H-optimus-1", pretrained=True, **timm_kwargs)
            except Exception:
                traceback.print_exc()
                raise Exception(
                    "Failed to download HOptimus-1 model, make sure that you were granted access "
                    "and that you correctly registered your token"
                )

        eval_transform = transforms.Compose(
            [
            transforms.Resize(224),  
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

        self._setup_feature_selector(
            model,
            feature_layer=feature_layer,
            block_index=block_index,
            block_indices=block_indices,
            fusion=fusion,
            token_pool=token_pool,
        )
        return model, eval_transform, torch.float16
    
    def forward(self, x):
        if getattr(self, "_selector_enabled", False):
            return self._extract_selected_vit_feature(x)
        return self.model(x)


encoder_registry = {
    "conch_v1": Conchv1InferenceEncoder,
    "uni_v2": UNIv2InferenceEncoder,
    "virchow2": Virchow2InferenceEncoder,
    "hoptimus1": HOptimus1InferenceEncoder,
}
