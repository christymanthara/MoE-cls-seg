"""
Stage 2 confidence-aware multimodal YOLO MoE inference.

Expected artifacts directory:
    router_confidence_aware.pt
    classification_best.pt
    segmentation_best.pt

Install:
    pip install torch open_clip_torch ultralytics pillow numpy

Decision IDs:
    0 = classification expert
    1 = segmentation expert
    2 = clarify / abstain (not an expert)
"""
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
from ultralytics import YOLO

DECISION_CLASSIFICATION = 0
DECISION_SEGMENTATION = 1
DECISION_CLARIFY = 2
DECISION_NAMES = {
    0: "classification",
    1: "segmentation",
    2: "clarify",
}


def get_ultralytics_device(device: torch.device):
    """Convert a PyTorch device into an Ultralytics inference device value."""
    if device.type == "cuda":
        return 0
    if device.type == "mps":
        return "mps"
    return "cpu"


class CLIPConfidenceRouter(nn.Module):
    """Frozen CLIP image/text encoder and a trainable three-decision routing MLP.

    forward() inputs:
        images: RGB float tensor [B,3,224,224] with values in [0,1].
        prompts: list of B natural-language instructions.

    forward() output:
        dict with logits [B,3], softmax probabilities [B,3], and decision IDs.
    """
    def __init__(self, clip_model_name="ViT-B-32",
                 clip_pretrained="laion2b_s34b_b79k", hidden_dim=256, dropout=0.1):
        super().__init__()
        self.clip_model_name = clip_model_name
        self.clip_pretrained = clip_pretrained
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model_name, pretrained=clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(clip_model_name)
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False
        self.clip_model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feature_dim = self.clip_model.encode_image(
                (dummy - self.clip_mean) / self.clip_std
            ).shape[-1]
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def train(self, mode=True):
        """Set gate mode while ensuring the frozen CLIP backbone remains in eval mode."""
        super().train(mode)
        self.clip_model.eval()
        return self

    def forward(self, images: torch.Tensor, prompts: List[str]):
        """Compute image/text features and return three-way routing logits."""
        images = (images - self.clip_mean.to(images.device)) / self.clip_std.to(images.device)
        tokens = self.tokenizer(prompts).to(images.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            text_features = self.clip_model.encode_text(tokens)
        image_features = F.normalize(image_features.float(), dim=-1)
        text_features = F.normalize(text_features.float(), dim=-1)
        fused = torch.cat(
            [image_features, text_features, image_features * text_features], dim=-1
        )
        logits = self.gate(fused)
        probs = torch.softmax(logits, dim=-1)
        return {"logits": logits, "probs": probs, "decision_id": probs.argmax(dim=-1)}


class ConfidenceAwareYOLOMoE:
    """Dispatch to YOLO experts only when the calibrated gate accepts the request.

    predict() accepts a file path, PIL image, or RGB float tensor [3,H,W] in [0,1].
    It returns routing probabilities and either an abstention result or the selected
    expert output. Expert confidence is task-specific and must not be compared across
    classification and segmentation.
    """
    def __init__(self, router, cls_weights_path, seg_weights_path,
                 temperature=1.0, routing_threshold=0.70,
                 cls_conf_threshold=0.50, seg_conf_threshold=0.50,
                 device=None):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.router = router.to(self.device).eval()
        self.cls_expert = YOLO(str(cls_weights_path))
        self.seg_expert = YOLO(str(seg_weights_path))
        self.temperature = float(temperature)
        self.routing_threshold = float(routing_threshold)
        self.cls_conf_threshold = float(cls_conf_threshold)
        self.seg_conf_threshold = float(seg_conf_threshold)

    @staticmethod
    def _to_pil(image):
        """Convert image path, PIL image, or [3,H,W] RGB float tensor to PIL."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if torch.is_tensor(image) and image.ndim == 3 and image.shape[0] == 3:
            array = (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                     * 255).round().astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        raise TypeError("image must be a path, PIL image, or [3,H,W] tensor")

    @torch.inference_mode()
    def predict(self, image, prompt: str, yolo_conf: float = 0.25):
        """Make a calibrated routing decision, abstain if needed, then call one expert."""
        pil = self._to_pil(image)
        resized = pil.resize((224, 224), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)

        output = self.router(tensor, [prompt])
        probabilities = torch.softmax(output["logits"] / self.temperature, dim=-1)[0]
        decision_id = int(probabilities.argmax().item())
        gate_confidence = float(probabilities.max().item())
        probability_dict = {
            DECISION_NAMES[i]: float(probabilities[i].item()) for i in range(3)
        }

        if decision_id == DECISION_CLARIFY:
            return {
                "status": "abstained", "route": "clarify",
                "reason": "The gate classified the instruction as ambiguous.",
                "gate_confidence": gate_confidence,
                "routing_probabilities": probability_dict,
                "needs_review": True, "result": None,
            }

        if gate_confidence < self.routing_threshold:
            return {
                "status": "abstained", "route": "clarify",
                "reason": (
                    f"Gate confidence {gate_confidence:.3f} is below threshold "
                    f"{self.routing_threshold:.3f}."
                ),
                "gate_confidence": gate_confidence,
                "routing_probabilities": probability_dict,
                "needs_review": True, "result": None,
            }

        yolo_device = get_ultralytics_device(self.device)
        if decision_id == DECISION_CLASSIFICATION:
            result = self.cls_expert.predict(
                pil, device=yolo_device, conf=yolo_conf, verbose=False
            )[0]
            if result.probs is None:
                raise RuntimeError("Classification expert returned no class probabilities.")
            expert_confidence = float(result.probs.top1conf.item())
            top_id = int(result.probs.top1)
            low_conf = expert_confidence < self.cls_conf_threshold
            return {
                "status": "completed", "route": "classification",
                "gate_confidence": gate_confidence,
                "routing_probabilities": probability_dict,
                "expert_confidence": expert_confidence,
                "top1_class_id": top_id,
                "top1_class_name": result.names[top_id],
                "needs_review": low_conf,
                "review_reason": (
                    "Classification score below configured threshold." if low_conf else None
                ),
                "result": result,
            }

        result = self.seg_expert.predict(
            pil, device=yolo_device, conf=yolo_conf, verbose=False
        )[0]
        scores = (
            result.boxes.conf.detach().cpu().tolist()
            if result.boxes is not None and len(result.boxes) else []
        )
        expert_confidence = max(scores) if scores else 0.0
        has_masks = result.masks is not None
        low_conf = expert_confidence < self.seg_conf_threshold
        needs_review = (not has_masks) or low_conf
        if not has_masks:
            review_reason = "No segmentation mask was returned."
        elif low_conf:
            review_reason = "Segmentation instance score below configured threshold."
        else:
            review_reason = None
        return {
            "status": "completed", "route": "segmentation",
            "gate_confidence": gate_confidence,
            "routing_probabilities": probability_dict,
            "expert_confidence": float(expert_confidence),
            "num_instances": len(scores), "has_masks": has_masks,
            "needs_review": needs_review, "review_reason": review_reason,
            "result": result,
        }


def load_confidence_moe(artifact_dir="./artifacts", device=None):
    """Load the Stage 2 gate and the two Stage 1 YOLO expert checkpoints.

    Inputs:
        artifact_dir: directory containing router_confidence_aware.pt and both YOLO .pt files.
        device: optional torch.device or string; automatically selected when omitted.
    Output:
        ready-to-use ConfidenceAwareYOLOMoE instance.
    """
    folder = Path(artifact_dir)
    checkpoint = torch.load(
        folder / "router_confidence_aware.pt", map_location="cpu"
    )
    router = CLIPConfidenceRouter(
        clip_model_name=checkpoint["clip_model_name"],
        clip_pretrained=checkpoint["clip_pretrained"],
        hidden_dim=checkpoint.get("hidden_dim", 256),
        dropout=checkpoint.get("dropout", 0.1),
    )
    router.load_state_dict(checkpoint["model_state_dict"])
    return ConfidenceAwareYOLOMoE(
        router=router,
        cls_weights_path=folder / "classification_best.pt",
        seg_weights_path=folder / "segmentation_best.pt",
        temperature=checkpoint["temperature"],
        routing_threshold=checkpoint["routing_threshold"],
        cls_conf_threshold=checkpoint.get("cls_conf_threshold", 0.50),
        seg_conf_threshold=checkpoint.get("seg_conf_threshold", 0.50),
        device=device,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Stage 2 confidence-aware YOLO MoE.")
    parser.add_argument("--artifacts", default="./artifacts")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default=None, help="Optional: cpu, mps, or cuda")
    args = parser.parse_args()
    model = load_confidence_moe(args.artifacts, device=args.device)
    prediction = model.predict(args.image, args.prompt)
    print({key: value for key, value in prediction.items() if key != "result"})
