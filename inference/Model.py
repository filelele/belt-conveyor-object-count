import sys
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import time

# Add model path to sys.path so mobileone can be imported cleanly
_MOBILEONE_DIR = Path(__file__).resolve().parent.parent / "model" / "ml-mobileone"
if str(_MOBILEONE_DIR) not in sys.path:
    sys.path.append(str(_MOBILEONE_DIR))

from mobileone import mobileone


class Model:
    """
    Wrapper around MobileOne for real-time inference on conveyor belt frames.
    """

    DEFAULT_CLASSES: List[str] = [
        "dilmah",
        "g7",
        "jack-jill",
        "karo",
        "nestea_atiso",
        "nestea_chanh",
        "nestea_hoaqua",
        "orion",
        "tipo",
        "y40",
    ]

    def __init__(
        self,
        checkpoint_path: str,
        variant: str = "s4",
        class_names: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        :param checkpoint_path: Path to the .pth or .pth.tar model weights.
        :param variant: MobileOne variant string ('s0', 's1', 's2', 's3', 's4').
        :param class_names: List of class name strings in index order (0 to N-1).
        :param device: 'cuda', 'cpu', or None (auto-detect).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.variant = variant

        # 1. Load checkpoint file
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        # Strip potential 'module.' prefixes from DDP training
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 2. Inspect the linear layer weight shape in the checkpoint
        # checkpoint weight shape is (out_features, in_features)
        num_classes = len(self.class_names)
        if "linear.weight" in state_dict:
            ckpt_classes = state_dict["linear.weight"].shape[0]
            assert ckpt_classes == num_classes, (
                f"Mismatch: Checkpoint has {ckpt_classes} classes, "
                f"but class_names has {num_classes}!"
            )

        # Detect if checkpoint is already reparameterized/fused or unfused
        is_reparam = not any("rbr_conv" in k for k in state_dict.keys())
        self.model = mobileone(variant=self.variant, num_classes=num_classes, inference_mode=is_reparam)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 6. Preprocessing pipeline (ImageNet normalization matching training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def process_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Processes a BGR/RGB numpy array frame, passes it through the model,
        and returns the predicted class label string.
        
        :param frame: Numpy array of the frame (e.g., from OpenCV or crop).
        :return: Predicted class string (e.g., 'orion', 'karo', etc.).
        """
        if frame is None or frame.size == 0:
            return ""

        # Convert OpenCV BGR format to RGB PIL Image if 3-channel
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Check if frame is BGR from OpenCV, convert to RGB
            pil_img = Image.fromarray(frame[:, :, ::-1])
        else:
            pil_img = Image.fromarray(frame)

        start_time = time.perf_counter()
        # Transform and add batch dimension -> (1, 3, 224, 224)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
            pred_idx = outputs.argmax(dim=1).item()
            end_time = time.perf_counter()
            return self.class_names[pred_idx], end_time - start_time
