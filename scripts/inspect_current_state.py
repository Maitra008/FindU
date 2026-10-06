"""
Inspect Current State of FindU Database, FAISS Index, and Reference Photos.
"""
import sys
import os
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.index import get_face_index

def main():
    print("=" * 70)
    print("FINDU — CURRENT STATE & REGISTERED IDENTITY INSPECTION")
    print("=" * 70)

    # 1. Database Inspection
    db_path = PROJECT_ROOT / "data" / "missing_person.db"
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        rows = c.execute("SELECT person_id, name, case_id, age, gender, status, photo_paths, embedding_path, is_active FROM persons").fetchall()
        print(f"\n[1] SQLite Database ({db_path}):")
        print(f"  Total Persons in DB: {len(rows)}")
        for r in rows:
            print(f"  - person_id     : {r[0]}")
            print(f"    name          : {r[1]}")
            print(f"    case_id       : {r[2]}")
            print(f"    age / gender  : {r[3]} / {r[4]}")
            print(f"    status / active: {r[5]} / {r[8]}")
            print(f"    photo_paths   : {r[6]}")
            print(f"    embedding_path: {r[7]}")
    else:
        print(f"[1] Database file NOT FOUND at {db_path}")

    # 2. Disk Reference Photos
    ref_base = PROJECT_ROOT / "data" / "reference_photos"
    print(f"\n[2] Reference Photos Directory ({ref_base}):")
    if ref_base.exists():
        subdirs = [d for d in ref_base.iterdir() if d.is_dir()]
        print(f"  Total Identity Subdirectories: {len(subdirs)}")
        for d in subdirs:
            photos = list(d.glob("*.*"))
            print(f"  - Directory: {d.name} ({len(photos)} files)")
            for p in photos:
                print(f"      * {p.name} ({p.stat().st_size} bytes)")
    else:
        print("  Directory does not exist.")

    # 3. Embedding Files on Disk
    emb_dir = PROJECT_ROOT / "models" / "embeddings"
    print(f"\n[3] Embedding Files ({emb_dir}):")
    if emb_dir.exists():
        emb_files = sorted(list(emb_dir.glob("*.npz")))
        print(f"  Total .npz files on disk: {len(emb_files)}")
        for f in emb_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    else:
        print("  Directory does not exist.")

    # 4. Canonical FAISS Index
    face_index = get_face_index()
    print(f"\n[4] Canonical FAISS Index (In-Memory & Persistent Metadata):")
    print(f"  Total Identities in FAISS Index (ntotal): {face_index.total_identities}")
    print(f"  Total Metadata Entries                 : {len(face_index.metadata)}")
    for m in face_index.metadata:
        print(f"  - Index #{m.get('index_id')}: person_id='{m.get('person_id')}', name='{m.get('name')}', file='{m.get('npz_file')}'")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
