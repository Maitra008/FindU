"""
Registration pipeline for Recognition Core.
Processes registration images for a person, extracts ArcFace embeddings,
computes a normalized identity template, and saves to disk (.npz).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from src.face_engine import FaceEngine, normalize_embedding

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class RegistrationResult:
    """Result of a person registration process."""
    person_id: str
    name: str
    num_images_processed: int
    num_valid_embeddings: int
    identity_embedding: np.ndarray
    output_path: Path


class FaceRegistrar:
    """
    Handles registration of new identities from 3-5 images.
    """

    def __init__(self, face_engine: Optional[FaceEngine] = None):
        """
        Initialize Registrar.

        Args:
            face_engine: Existing FaceEngine instance, or creates a new one if None.
        """
        self.engine = face_engine or FaceEngine()

    def register_person(
        self,
        person_id: str,
        name: str,
        image_dir_or_paths: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path] = Path("models/embeddings"),
    ) -> Optional[RegistrationResult]:
        """
        Process registration images for a person and generate an identity embedding.

        Pipeline:
        1. Read each image with OpenCV
        2. SCRFD face detection
        3. Face selection:
           - exactly 1 face: use it
           - 0 faces: log warning and skip
           - >1 faces: log warning and select highest-confidence face
        4. Extract ArcFace embedding & L2-normalize individually
        5. Mean of all valid normalized embeddings
        6. L2-normalize the mean embedding
        7. Save to .npz file

        Args:
            person_id: Unique string identifier (e.g. 'person_x').
            name: Full name of the person (e.g. 'Person X').
            image_dir_or_paths: Directory containing registration images or list of image paths.
            output_dir: Directory where the .npz identity file will be saved.

        Returns:
            RegistrationResult if registration succeeds, or None if no valid faces found.
        """
        output_path_dir = Path(output_dir)
        output_path_dir.mkdir(parents=True, exist_ok=True)

        # Collect image file paths
        image_files: List[Path] = []
        if isinstance(image_dir_or_paths, (str, Path)):
            dir_path = Path(image_dir_or_paths)
            if dir_path.is_dir():
                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.extend(dir_path.glob(f"*{ext}"))
                    image_files.extend(dir_path.glob(f"*{ext.upper()}"))
            elif dir_path.is_file():
                image_files.append(dir_path)
            else:
                logger.error("Registration path not found: %s", dir_path)
                return None
        elif isinstance(image_dir_or_paths, list):
            image_files = [Path(p) for p in image_dir_or_paths if Path(p).is_file()]

        image_files = sorted(list(set(image_files)))
        logger.info("Found %d registration image(s) for person_id='%s'", len(image_files), person_id)

        if not image_files:
            logger.error("No valid image files found in %s", image_dir_or_paths)
            return None

        valid_embeddings: List[np.ndarray] = []
        successful_images: List[str] = []

        for img_path in image_files:
            logger.info("Processing registration image: %s", img_path.name)
            img = self.engine.load_image(img_path)
            if img is None:
                logger.warning("Could not load image: %s", img_path)
                continue

            detected_faces = self.engine.detect_and_embed(img)

            if len(detected_faces) == 0:
                logger.warning("No face detected in registration image: %s. Skipping.", img_path.name)
                continue
            elif len(detected_faces) == 1:
                selected_face = detected_faces[0]
                logger.info(
                    "Single face detected in %s (confidence: %.4f)",
                    img_path.name,
                    selected_face.confidence,
                )
            else:
                # Multiple faces found: Log warning and select highest-confidence face
                selected_face = max(detected_faces, key=lambda f: f.confidence)
                logger.warning(
                    "Multiple faces (%d) detected in %s. Selecting highest confidence face (conf: %.4f, bbox: %s).",
                    len(detected_faces),
                    img_path.name,
                    selected_face.confidence,
                    selected_face.bbox,
                )

            # Individual embedding is already L2-normalized by FaceEngine, but enforce strictly
            norm_emb = normalize_embedding(selected_face.normalized_embedding)
            valid_embeddings.append(norm_emb)
            successful_images.append(img_path.name)

        if not valid_embeddings:
            logger.error("Registration failed: 0 valid face embeddings extracted for person_id='%s'", person_id)
            return None

        logger.info(
            "Extracted %d valid embeddings out of %d images for person_id='%s'",
            len(valid_embeddings),
            len(image_files),
            person_id,
        )

        # 5. Average normalized embeddings
        embeddings_matrix = np.vstack(valid_embeddings)  # shape (N, 512)
        mean_embedding = np.mean(embeddings_matrix, axis=0)

        # 6. L2-normalize the resulting mean embedding
        identity_embedding = normalize_embedding(mean_embedding)

        # 7. Save to .npz file
        save_file = output_path_dir / f"{person_id}_embedding.npz"
        np.savez_compressed(
            save_file,
            person_id=person_id,
            name=name,
            num_images=len(valid_embeddings),
            embedding=identity_embedding,
            image_names=np.array(successful_images),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        logger.info("Saved identity embedding to: %s", save_file)

        return RegistrationResult(
            person_id=person_id,
            name=name,
            num_images_processed=len(image_files),
            num_valid_embeddings=len(valid_embeddings),
            identity_embedding=identity_embedding,
            output_path=save_file,
        )

