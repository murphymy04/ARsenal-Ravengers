"""Face embedding extraction using EdgeFace.

Loads the EdgeFace model locally (no torch.hub) and extracts
512-dimensional face embeddings from 112x112 RGB crops.
"""

import sys
import numpy as np
import torch
from torchvision import transforms

from models import FaceEmbedding
from config import EDGEFACE_ROOT, EDGEFACE_CHECKPOINT, EMBEDDING_MODEL_NAME


class FaceEmbedder:
    """Extracts 512-dim face embeddings using EdgeFace."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        checkpoint_path: str = str(EDGEFACE_CHECKPOINT),
        device: str = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load EdgeFace model from local source
        edgeface_root = str(EDGEFACE_ROOT)
        if edgeface_root not in sys.path:
            sys.path.insert(0, edgeface_root)

        from backbones import get_model

        self._model = get_model(model_name)
        self._model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        )
        self._model.to(self.device)
        self._model.eval()

        # Same preprocessing as face_demo.py
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def embed(self, crop: np.ndarray) -> FaceEmbedding:
        """Extract embedding from a 112x112 RGB face crop.

        Args:
            crop: 112x112 RGB uint8 numpy array.

        Returns:
            FaceEmbedding with 512-dim vector.
        """
        tensor = self._transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self._model(tensor)
        return FaceEmbedding(
            vector=vector.cpu().numpy()[0],
            model_name=self.model_name,
        )

    def embed_batch(self, crops: list[np.ndarray]) -> list[FaceEmbedding]:
        """Extract embeddings for multiple crops in a single batch.

        Args:
            crops: list of 112x112 RGB uint8 numpy arrays.

        Returns:
            List of FaceEmbedding objects.
        """
        if not crops:
            return []
        tensors = torch.stack([self._transform(c) for c in crops]).to(self.device)
        with torch.no_grad():
            vectors = self._model(tensors)
        vectors_np = vectors.cpu().numpy()
        return [
            FaceEmbedding(vector=v, model_name=self.model_name)
            for v in vectors_np
        ]
