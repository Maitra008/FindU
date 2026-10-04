"""
CLI Entry Point for Recognition Core.
Provides subcommands:
  - register:    Register a new person from 3-5 images and generate identity embedding.
  - build-index: Index all registered identity embeddings into FAISS.
  - recognize:   Process a video file or live webcam with SCRFD, ArcFace, and FAISS match.
  - evaluate:    Evaluate recognition performance across thresholds and produce reports/plots.
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import RecognitionConfig
from src.evaluation import ModelEvaluator, print_full_evaluation_report
from src.face_engine import FaceEngine
from src.index import FaceIndex
from src.recognition import VideoRecognizer
from src.registration import FaceRegistrar


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging output."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def handle_register(args: argparse.Namespace, config: RecognitionConfig) -> int:
    """Handle person registration command."""
    print("Loading InsightFace...")
    engine = FaceEngine(
        model_name=config.model_name,
        providers=config.providers,
        det_size=config.det_size,
        det_thresh=config.det_thresh,
    )
    print("Model loaded.\n")

    registrar = FaceRegistrar(face_engine=engine)
    output_dir = Path(args.output_dir) if args.output_dir else config.embeddings_dir

    result = registrar.register_person(
        person_id=args.person_id,
        name=args.name,
        image_dir_or_paths=args.images,
        output_dir=output_dir,
    )

    if result is None:
        print(f"\n[ERROR] Registration failed for person '{args.person_id}'.")
        return 1

    print(f"\n[SUCCESS] Successfully registered '{result.name}' ({result.person_id})")
    print(f"  Valid images: {result.num_valid_embeddings}/{result.num_images_processed}")
    print(f"  Embedding saved to: {result.output_path}")
    print("\nNext step: Run 'python -m src.main build-index' to update the FAISS search index.")
    return 0


def handle_build_index(args: argparse.Namespace, config: RecognitionConfig) -> int:
    """Handle FAISS index generation from saved embeddings."""
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else config.embeddings_dir
    index_path = Path(args.index_path) if args.index_path else config.index_path
    metadata_path = Path(args.metadata_path) if args.metadata_path else config.metadata_path

    print(f"Scanning embeddings in: {embeddings_dir}")
    index = FaceIndex(dimension=config.embedding_dim)
    count = index.build_from_directory(embeddings_dir)

    if count == 0:
        print(f"[WARNING] No identity embeddings found in {embeddings_dir}. Built an empty index.")
    else:
        index.save(index_path, metadata_path)
        print(f"\n[SUCCESS] Indexed {count} identity/identities into FAISS.")
        print(f"  Index file:    {index_path}")
        print(f"  Metadata file: {metadata_path}")
    return 0


def handle_recognize(args: argparse.Namespace, config: RecognitionConfig) -> int:
    """Handle video and live camera recognition command."""
    index_path = Path(args.index_path) if args.index_path else config.index_path
    metadata_path = Path(args.metadata_path) if args.metadata_path else config.metadata_path

    if not index_path.exists() or not metadata_path.exists():
        print(f"[ERROR] FAISS index or metadata not found at:\n  Index: {index_path}\n  Metadata: {metadata_path}")
        print("Please register identities and run 'python -m src.main build-index' first.")
        return 1

    face_index = FaceIndex(dimension=config.embedding_dim)
    if not face_index.load(index_path, metadata_path):
        print(f"[ERROR] Failed to load FAISS index from {index_path}")
        return 1

    print(f"Loaded FAISS index with {face_index.total_identities} registered identities.")

    print("Loading InsightFace...")
    engine = FaceEngine(
        model_name=config.model_name,
        providers=config.providers,
        det_size=config.det_size,
        det_thresh=config.det_thresh,
    )
    print("Model loaded.\n")

    threshold = args.threshold if args.threshold is not None else config.threshold
    sample_fps = args.sample_fps if args.sample_fps is not None else config.sample_fps
    iou_thresh = getattr(args, "iou_threshold", config.iou_threshold)
    max_missed = getattr(args, "max_missed_frames", config.max_missed_frames)

    recognizer = VideoRecognizer(
        face_engine=engine,
        face_index=face_index,
        threshold=threshold,
        sample_fps=sample_fps,
        verbose_debug=args.debug,
        iou_threshold=iou_thresh,
        max_missed_frames=max_missed,
    )

    try:
        if args.camera is not None:
            recognizer.process_camera(
                camera_index=args.camera,
                threshold=threshold,
                sample_fps=sample_fps,
                display=True,
                iou_threshold=iou_thresh,
                max_missed_frames=max_missed,
            )
        else:
            recognizer.process_video(
                video_path=args.video,
                threshold=threshold,
                sample_fps=sample_fps,
                display=getattr(args, "display", False),
                iou_threshold=iou_thresh,
                max_missed_frames=max_missed,
            )
        return 0
    except Exception as e:
        print(f"\n[ERROR] Recognition failed: {e}")
        logging.exception("Exception in recognition processing")
        return 1


def handle_evaluate(args: argparse.Namespace, config: RecognitionConfig) -> int:
    """Handle dataset evaluation and threshold tuning command."""
    print("Loading InsightFace...")
    engine = FaceEngine(
        model_name=config.model_name,
        providers=config.providers,
        det_size=config.det_size,
        det_thresh=config.det_thresh,
    )
    print("Model loaded.\n")

    evaluator = ModelEvaluator(face_engine=engine)
    
    evaluation_dir = Path(args.dataset)
    registration_dir = Path(args.registration) if args.registration else config.registration_dir
    output_dir = Path(args.output) if args.output else Path("data/evaluation/results")
    target_max_fpr = args.target_max_fpr

    print(f"Running evaluation...")
    print(f"  Evaluation dataset:   {evaluation_dir}")
    print(f"  Registration dataset: {registration_dir}")
    print(f"  Results output:       {output_dir}")
    print()

    try:
        report = evaluator.run_full_evaluation(
            dataset_dir=evaluation_dir,
            validation_dir=getattr(args, "validation", None),
            test_dir=getattr(args, "test", None),
            registration_dir=registration_dir,
            output_dir=output_dir,
            target_max_fpr=target_max_fpr,
        )
        print_full_evaluation_report(report)
        return 0
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        logging.exception("Exception during evaluation")
        return 1


def handle_serve(args: argparse.Namespace, config: RecognitionConfig) -> int:
    """Handle FastAPI server startup."""
    import uvicorn
    from src.db.database import init_db
    from src.workers.worker_manager import get_worker_manager

    print("Initializing database...")
    init_db(seed_defaults=True)

    if getattr(args, "start_workers", False):
        print("Starting camera workers...")
        worker_mgr = get_worker_manager()
        worker_mgr.load_cameras_from_db(enabled_only=True)
        worker_mgr.start_all()

    print(f"Starting API server on http://{args.host}:{args.port}")
    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False, log_level="info")
    return 0


def main() -> None:
    """CLI entry point."""
    config = RecognitionConfig()
    config.ensure_directories()

    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Recognition Core MVP - SCRFD & ArcFace Face Recognition Pipeline",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set logging level (default: WARNING)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: register
    parser_reg = subparsers.add_parser("register", help="Register a person from 3-5 images")
    parser_reg.add_argument("--person-id", required=True, help="Unique identifier (e.g., person_x)")
    parser_reg.add_argument("--name", required=True, help="Full display name (e.g., 'Person X')")
    parser_reg.add_argument("--images", required=True, help="Directory or path containing registration images")
    parser_reg.add_argument("--output-dir", default=str(config.embeddings_dir), help="Directory to save .npz identity")

    # Command: build-index
    parser_idx = subparsers.add_parser("build-index", help="Build FAISS index from stored identity embeddings")
    parser_idx.add_argument("--embeddings-dir", default=str(config.embeddings_dir), help="Directory containing .npz files")
    parser_idx.add_argument("--index-path", default=str(config.index_path), help="Path to write .faiss file")
    parser_idx.add_argument("--metadata-path", default=str(config.metadata_path), help="Path to write .json metadata")

    # Command: recognize
    parser_rec = subparsers.add_parser("recognize", help="Run face recognition on a video file or live webcam")
    source_group = parser_rec.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", help="Path to input video file")
    source_group.add_argument("--camera", type=int, help="Webcam device index (e.g., 0)")

    parser_rec.add_argument("--threshold", type=float, default=config.threshold, help="Similarity threshold (default: 0.89)")
    parser_rec.add_argument("--sample-fps", type=float, default=config.sample_fps, help="Sampling rate in FPS (default: 2.0)")
    parser_rec.add_argument("--iou-threshold", type=float, default=config.iou_threshold, help="IoU threshold for face tracking (default: 0.30)")
    parser_rec.add_argument("--max-missed-frames", type=int, default=config.max_missed_frames, help="Max missed frames before track termination (default: 5)")
    parser_rec.add_argument("--display", action="store_true", help="Display visual video feed window with bounding box overlays")
    parser_rec.add_argument("--index-path", default=str(config.index_path), help="Path to FAISS .faiss file")
    parser_rec.add_argument("--metadata-path", default=str(config.metadata_path), help="Path to metadata JSON file")
    parser_rec.add_argument("--debug", action="store_true", help="Print verbose frame/face debugging information (including unknown detections)")

    # Command: evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate recognition performance across thresholds")
    parser_eval.add_argument("--dataset", required=True, help="Path to evaluation dataset directory")
    parser_eval.add_argument("--registration", default=str(config.registration_dir), help="Path to registration dataset directory")
    parser_eval.add_argument("--output", default="data/evaluation/results", help="Directory to save evaluation results and plots")
    parser_eval.add_argument("--target-max-fpr", type=float, default=0.05, help="Target max FPR constraint for threshold selection (default: 0.05)")
    parser_eval.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")

    # Command: serve (Part 4)
    parser_srv = subparsers.add_parser("serve", help="Start FastAPI REST & WebSocket server")
    parser_srv.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser_srv.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser_srv.add_argument("--start-workers", action="store_true", help="Auto-start camera workers on boot")

    args = parser.parse_args()

    configure_logging(args.log_level)

    if args.command == "register":
        sys.exit(handle_register(args, config))
    elif args.command == "build-index":
        sys.exit(handle_build_index(args, config))
    elif args.command == "recognize":
        sys.exit(handle_recognize(args, config))
    elif args.command == "evaluate":
        sys.exit(handle_evaluate(args, config))
    elif args.command == "serve":
        sys.exit(handle_serve(args, config))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
