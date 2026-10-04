"""
FAISS Index management module for cosine similarity search over identity embeddings.
Uses faiss.IndexFlatIP for exact inner-product search on L2-normalized vectors.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

from src.face_engine import normalize_embedding

logger = logging.getLogger(__name__)


@dataclass
class IdentityMetadata:
    """Metadata associated with an indexed identity."""
    index_id: int
    person_id: str
    name: str
    num_images: int
    npz_file: str


@dataclass
class MatchResult:
    """Result of a similarity search query against the FAISS index."""
    person_id: Optional[str]
    name: Optional[str]
    similarity: float
    is_match: bool
    index_id: int


class FaceIndex:
    """
    FAISS-backed identity index using IndexFlatIP for exact cosine similarity search.
    """

    def __init__(self, dimension: int = 512):
        """
        Initialize an empty FAISS IndexFlatIP.

        Args:
            dimension: Dimensionality of embeddings (ArcFace uses 512).
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    @property
    def total_identities(self) -> int:
        """Return number of registered identities in the index."""
        return self.index.ntotal

    def add_identity(
        self,
        embedding: np.ndarray,
        person_id: str,
        name: str,
        num_images: int = 1,
        npz_file: str = "",
    ) -> int:
        """
        Add a single L2-normalized identity embedding to the index.

        Args:
            embedding: 1D or (1, D) numpy array.
            person_id: Unique identifier for the person.
            name: Human-readable name.
            num_images: Count of registration images used.
            npz_file: Path to source .npz file.

        Returns:
            The integer index assigned in FAISS.
        """
        emb = normalize_embedding(embedding)
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)

        if emb.shape[1] != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {emb.shape[1]}")

        idx = self.index.ntotal
        self.index.add(emb.astype(np.float32))

        meta = {
            "index_id": idx,
            "person_id": person_id,
            "name": name,
            "num_images": num_images,
            "npz_file": str(npz_file),
        }
        self.metadata.append(meta)
        logger.info("Added '%s' (%s) at index %d. Total index size: %d", name, person_id, idx, self.index.ntotal)
        return idx

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 1,
        threshold: float = 0.89,
    ) -> List[MatchResult]:
        """
        Search FAISS index for top-k closest matches for a normalized query embedding.

        Args:
            query_embedding: (512,) or (N, 512) float32 query array.
            k: Top-k nearest neighbours to return.
            threshold: Cosine similarity cutoff.

        Returns:
            List of MatchResult objects.
        """
        if self.index.ntotal == 0:
            return [
                MatchResult(
                    person_id=None,
                    name="Unknown",
                    similarity=0.0,
                    is_match=False,
                    index_id=-1,
                )
            ]

        query = normalize_embedding(query_embedding)
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)

        # faiss.IndexFlatIP returns inner products (cosine similarities for unit vectors)
        similarities, indices = self.index.search(query.astype(np.float32), min(k, self.index.ntotal))

        results: List[MatchResult] = []
        for sim, idx in zip(similarities[0], indices[0]):
            sim_val = float(sim)
            if idx >= 0 and idx < len(self.metadata):
                meta = self.metadata[idx]
                is_match = sim_val >= threshold
                results.append(
                    MatchResult(
                        person_id=meta["person_id"] if is_match else None,
                        name=meta["name"] if is_match else "Unknown",
                        similarity=sim_val,
                        is_match=is_match,
                        index_id=int(idx),
                    )
                )
            else:
                results.append(
                    MatchResult(
                        person_id=None,
                        name="Unknown",
                        similarity=sim_val,
                        is_match=False,
                        index_id=-1,
                    )
                )

        return results

    def build_from_directory(self, embeddings_dir: Union[str, Path]) -> int:
        """
        Scan a directory for *_embedding.npz files and build a clean FAISS index.

        Args:
            embeddings_dir: Directory containing .npz identity files.

        Returns:
            Number of indexed identities.
        """
        dir_path = Path(embeddings_dir)
        if not dir_path.exists():
            logger.warning("Embeddings directory does not exist: %s", dir_path)
            return 0

        npz_files = sorted(list(dir_path.glob("*_embedding.npz")) + list(dir_path.glob("*.npz")))
        npz_files = sorted(list(set(npz_files)))

        logger.info("Found %d embedding file(s) in %s", len(npz_files), dir_path)

        # Reset index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

        for npz_path in npz_files:
            try:
                data = np.load(str(npz_path), allow_pickle=True)
                emb = data["embedding"]
                person_id = str(data["person_id"])
                name = str(data["name"])
                num_images = int(data.get("num_images", 1))

                self.add_identity(
                    embedding=emb,
                    person_id=person_id,
                    name=name,
                    num_images=num_images,
                    npz_file=str(npz_path),
                )
            except Exception as e:
                logger.error("Failed to load identity file '%s': %s", npz_path, e)

        logger.info("FAISS index built with %d identities.", self.index.ntotal)
        return self.index.ntotal

    def save(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        """
        Serialize FAISS index and metadata mapping to disk.

        Args:
            index_path: Path to write the .faiss file.
            metadata_path: Path to write the .json metadata file.
        """
        index_p = Path(index_path)
        meta_p = Path(metadata_path)

        index_p.parent.mkdir(parents=True, exist_ok=True)
        meta_p.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_p))
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved FAISS index to %s and metadata to %s", index_p, meta_p)

    def load(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> bool:
        """
        Load FAISS index and metadata mapping from disk.

        Args:
            index_path: Path to the .faiss file.
            metadata_path: Path to the .json metadata file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        index_p = Path(index_path)
        meta_p = Path(metadata_path)

        if not index_p.exists() or not meta_p.exists():
            logger.error("Index file (%s) or metadata file (%s) does not exist.", index_p, meta_p)
            return False

        try:
            self.index = faiss.read_index(str(index_p))
            with open(meta_p, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            logger.info("Loaded FAISS index with %d identities from %s", self.index.ntotal, index_p)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index / metadata: %s", e)
            return False



_face_index_instance: Optional[FaceIndex] = None


def get_face_index(dimension: int = 512) -> FaceIndex:
    """Singleton getter for the shared dynamic FAISS FaceIndex."""
    global _face_index_instance
    if _face_index_instance is None:
        _face_index_instance = FaceIndex(dimension=dimension)
        from src.config import EMBEDDINGS_DIR, INDEX_PATH, METADATA_PATH
        if INDEX_PATH.exists() and METADATA_PATH.exists():
            _face_index_instance.load(INDEX_PATH, METADATA_PATH)
        elif EMBEDDINGS_DIR.exists():
            _face_index_instance.build_from_directory(EMBEDDINGS_DIR)
    return _face_index_instance
