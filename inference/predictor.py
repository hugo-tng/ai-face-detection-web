import os
import torch
import base64
from io import BytesIO
from PIL import Image

from inference.detector import DeepfakeDetector
from inference.facecrop import FaceCropper
from inference.datasets import get_transforms
from inference.config import GlobalConfig, LabelConfig


class DeepfakePredictor:
    """
    Inference-only predictor for DeepfakeDetector.
    Model is initialized once and reused for all requests.
    """

    def __init__(
        self,
        model_dir: str,
        mode: str = "hybrid",
        img_size: int = 224,
        efficientnet_model: str = "efficientnet_b1",
        spatial_dim: int = 512,
        freq_dim: int = 256,
        use_attention: bool = True,
    ):
        self.device = GlobalConfig.DEVICE
        self.img_size = img_size
        self.mode = mode

        # ===============================
        # 1️⃣ Build model architecture
        # ===============================
        self.model = DeepfakeDetector(
            mode=mode,
            num_classes=2,
            img_size=img_size,
            efficientnet_model=efficientnet_model,
            spatial_dim=spatial_dim,
            freq_dim=freq_dim,
            use_attention_fusion=use_attention,
        ).to(self.device)

        # ===============================
        # 2️⃣ Load checkpoint
        # ===============================
        ckpt_path = os.path.join(model_dir, "best_model.pth")
        self._load_weights(ckpt_path)

        self.model.eval()
        print("✅ DeepfakeDetector loaded and ready for inference")

        # ===============================
        # 3️⃣ Preprocessing utilities
        # ===============================
        self.face_cropper = FaceCropper(
            out_size=GlobalConfig.CROP_SIZE,
            target_face_ratio=1.0,
            scale_factor=1.1,
            min_neighbors=4
        )
        self.transform = get_transforms(img_size)["test"]

    # --------------------------------------------------
    # Internal utilities
    # --------------------------------------------------

    def _load_weights(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")

        print(f"🔄 Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if trained with DataParallel
        clean_state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }

        self.model.load_state_dict(clean_state_dict, strict=False)

    # --------------------------------------------------
    # Public inference APIs
    # --------------------------------------------------

    @torch.no_grad()
    def predict_from_pil(self, pil_image: Image.Image):
        """
        Inference directly from PIL Image (no disk I/O).
        """
        # 1️⃣ Face crop
        cropped = self.face_cropper(pil_image)

        # 2️⃣ Transform → Tensor
        img_tensor = (
            self.transform(cropped)
            .unsqueeze(0)
            .to(self.device)
        )

        # 3️⃣ Forward
        logits = self.model(img_tensor)
        probs = torch.softmax(logits, dim=1)

        fake_prob = probs[0][LabelConfig.FAKE_IDX].item()
        real_prob = probs[0][LabelConfig.REAL_IDX].item()

        label = "FAKE" if fake_prob > real_prob else "REAL"

        # 4️⃣ Attention weights (optional)
        spatial_w, freq_w = None, None
        if hasattr(self.model, "get_feature_importance"):
            try:
                sw, fw = self.model.get_feature_importance(img_tensor)
                spatial_w = sw.mean().item()
                freq_w = fw.mean().item()
            except Exception:
                pass

        # 5️⃣ Encode preview image (base64 for HTML)
        buffer = BytesIO()
        cropped.save(buffer, format="PNG")
        preview_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "label": label,
            "fake_prob": round(fake_prob * 100, 2),
            "real_prob": round(real_prob * 100, 2),
            "spatial_weight": spatial_w,
            "frequency_weight": freq_w,
            "preview": preview_base64,
        }

    @torch.no_grad()
    def predict_from_bytes(self, image_bytes: bytes):
        """
        Inference from raw image bytes (useful for API).
        """
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.predict_from_pil(pil_img)
