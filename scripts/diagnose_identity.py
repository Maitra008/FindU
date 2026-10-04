"""
Read-Only Identity & FAISS Diagnostic Script for FindU.
Inspects the canonical FaceEngine and FaceIndex singletons, verifies identity registration
for 'partho_b55022', and performs a direct FAISS cosine similarity search using the stored template.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config import DEFAULT_SIMILARITY_THRESHOLD, EMBEDDINGS_DIR, MODELS_DIR
from src.face_engine import get_face_engine
from src.index import get_face_index


def run_diagnostic(target_person_id: str = "partho_b55022") -> None:
    print("=" * 70)
    print("FINDU — READ-ONLY IDENTITY & FAISS RECOGNITION DIAGNOSTIC")
    print("=" * 70)

    # 1. Obtain Canonical Singletons
    engine = get_face_engine()
    index = get_face_index()

    engine_id = id(engine)
    index_id = id(index)
    ntotal = index.total_identities
    meta_count = len(index.metadata)

    print(f"\n[CANONICAL INSTANCE IDENTIFIERS]")
    print(f"  FaceEngine Object ID : {engine_id}")
    print(f"  FaceIndex Object ID  : {index_id}")
    print(f"  FAISS Index (ntotal) : {ntotal} vectors")
    print(f"  Metadata Count       : {meta_count} identities")

    # 2. Check Identity in Metadata & FAISS
    matching_meta = [m for m in index.metadata if m.get("person_id") == target_person_id]
    meta_exists = len(matching_meta) > 0

    print(f"\n[TARGET IDENTITY SEARCH: '{target_person_id}']")
    print(f"  Found in FaceIndex Metadata : {'YES' if meta_exists else 'NO'}")
    if meta_exists:
        for idx_info in matching_meta:
            print(f"    - Index ID   : {idx_info.get('index_id')}")
            print(f"    - Display Name: {idx_info.get('name')}")
            print(f"    - Source .npz : {idx_info.get('npz_file')}")
            print(f"    - Image Count : {idx_info.get('num_images')}")

    # 3. Check Physical Paths on Disk
    ref_photo_dir = PROJECT_ROOT / "data" / "reference_photos" / target_person_id
    npz_candidate_1 = EMBEDDINGS_DIR / f"{target_person_id}_embedding.npz"
    npz_candidate_2 = EMBEDDINGS_DIR / f"{target_person_id}.npz"

    resolved_npz: Path | None = None
    if npz_candidate_1.exists():
        resolved_npz = npz_candidate_1
    elif npz_candidate_2.exists():
        resolved_npz = npz_candidate_2

    print(f"\n[FILE STORAGE VERIFICATION]")
    print(f"  Reference Photo Directory : {ref_photo_dir}")
    print(f"    - Exists on Disk        : {'YES' if ref_photo_dir.exists() else 'NO'}")
    if ref_photo_dir.exists():
        photos = list(ref_photo_dir.glob("*.*"))
        print(f"    - Total Photo Files     : {len(photos)}")
        for p in photos:
            print(f"        * {p.name} ({p.stat().st_size} bytes)")

    print(f"  Embedding File (.npz)     : {resolved_npz if resolved_npz else npz_candidate_1}")
    print(f"    - Exists on Disk        : {'YES' if resolved_npz else 'NO'}")

    registered_embedding: np.ndarray | None = None

    if resolved_npz and resolved_npz.exists():
        try:
            data = np.load(str(resolved_npz), allow_pickle=True)
            registered_embedding = data["embedding"]
            emb_shape = registered_embedding.shape
            emb_dtype = registered_embedding.dtype
            l2_norm = float(np.linalg.norm(registered_embedding))

            print(f"\n[EMBEDDING SPECIFICATIONS]")
            print(f"  Embedding Path : {resolved_npz}")
            print(f"  Shape          : {emb_shape}")
            print(f"  Dtype          : {emb_dtype}")
            print(f"  L2 Norm        : {l2_norm:.6f} {'(Strict Unit Vector)' if abs(l2_norm - 1.0) < 1e-4 else '(Unnormalized)'}")
        except Exception as e:
            print(f"  [ERROR] Failed to load embedding file: {e}")

    # 4. Direct FAISS Search using Registered Embedding
    if registered_embedding is not None and ntotal > 0:
        print(f"\n[DIRECT FAISS SIMILARITY QUERY -- TOP 5 MATCHES]")
        print(f"  Similarity Threshold : {DEFAULT_SIMILARITY_THRESHOLD}")
        print("-" * 70)
        print(f"{'Rank':<6} {'Person ID':<28} {'Similarity':<12} {'Threshold':<12} {'Status'}")
        print("-" * 70)

        # Prepare normalized query vector (1, 512) float32
        query_vec = registered_embedding.astype(np.float32)
        norm_val = np.linalg.norm(query_vec)
        if norm_val > 0:
            query_vec = query_vec / norm_val
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        k = min(5, ntotal)
        similarities, indices = index.index.search(query_vec, k)

        for rank, (sim, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
            sim_val = float(sim)
            if 0 <= idx < len(index.metadata):
                m = index.metadata[idx]
                pid = m.get("person_id", "Unknown")
                pname = m.get("name", "Unknown")
                display_name = f"{pname} ({pid})"
            else:
                display_name = f"Index #{idx}"

            is_match = sim_val >= DEFAULT_SIMILARITY_THRESHOLD
            status_str = "MATCH" if is_match else "NO MATCH"

            print(f"{rank:<6} {display_name:<28} {sim_val:<12.4f} {DEFAULT_SIMILARITY_THRESHOLD:<12.2f} {status_str}")

        print("-" * 70)
        top_sim = float(similarities[0][0])
        print(f"\n[SELF-MATCH EVALUATION]")
        print(f"  Top Match Cosine Similarity : {top_sim:.6f}")
        if top_sim >= 0.99:
            print(f"  Self-Match Result           : PERFECT (Near 1.0 identity self-similarity)")
        elif top_sim >= DEFAULT_SIMILARITY_THRESHOLD:
            print(f"  Self-Match Result           : PASS (Meets threshold >= {DEFAULT_SIMILARITY_THRESHOLD})")
        else:
            print(f"  Self-Match Result           : WARNING (Below threshold {DEFAULT_SIMILARITY_THRESHOLD})")
    elif ntotal == 0:
        print("\n[WARNING] FAISS index is currently empty (ntotal = 0).")
    else:
        print("\n[WARNING] No registered embedding available to execute search.")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "partho_b55022"
    run_diagnostic(target)
