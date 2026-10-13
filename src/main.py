"""
Unmayaverse: Multimodal Video Retrieval Pipeline CLI
Unified Entry Point for Ingestion, Indexing, Search, Evaluation & UI
"""

import os
import sys
import json
import argparse
from typing import Optional

from src.config import (
    MEDIA_FOLDER_PATH,
    QDRANT_DB_DIR,
    VIDEO_FILE,
    GOLDEN_DATASET_FILE,
    SEARCH_MODE_VISUAL,
    SEARCH_MODE_VISUAL_TEXT,
    SEARCH_MODE_VISUAL_TEXT_AUDIO,
    SEARCH_MODES,
    DEFAULT_FPS,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_REFERENCE_FRAME,
    DEFAULT_FACE_THRESHOLD,
    DEFAULT_TOP_K,
    MIN_RRF_SCORE,
    DEFAULT_GRADIO_PORT,
)


def get_default_device() -> str:
    """Auto-detects best available compute device (CUDA -> MPS -> CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def run_preprocess(args):
    """Executes end-to-end raw media extraction and embedding generation."""
    from src.preprocessor import DataPreprocessor

    media_folder = getattr(args, "media_folder", MEDIA_FOLDER_PATH)
    fps = getattr(args, "fps", DEFAULT_FPS)
    whisper_model = getattr(args, "whisper_model", DEFAULT_WHISPER_MODEL)
    ref_frame = getattr(args, "reference_frame", DEFAULT_REFERENCE_FRAME)
    face_threshold = getattr(args, "face_threshold", DEFAULT_FACE_THRESHOLD)
    video_path = getattr(args, "video_path", None) or os.path.join(media_folder, VIDEO_FILE)

    print(f"\n[1/6] Initializing Preprocessor for: {media_folder}")
    preprocessor = DataPreprocessor(media_folder_path=media_folder)

    if not os.path.exists(video_path):
        print(f"[Error] Video file not found at: {video_path}")
        return

    print(f"Processing video: {video_path}")
    print("\n[2/6] Extracting 1 FPS Frames...")
    preprocessor.extract_frames(video_file=video_path, fps=fps)

    print("\n[3/6] Extracting Audio & Separating Vocals/Score (Demucs)...")
    preprocessor.extract_master_audio(video_file=video_path)
    preprocessor.separate_audio_demucs()

    print("\n[4/6] Transcribing Speech Dialogue (Whisper)...")
    preprocessor.transcribe_audio_whisper(model_name=whisper_model)

    print("\n[5/6] Extracting Character Face Embeddings (InsightFace)...")
    preprocessor.extract_maya_reference_embedding(reference_input=ref_frame)
    preprocessor.run_face_analysis(similarity_threshold=face_threshold)
    preprocessor.extract_and_align_all_raw_data()

    print("\n[6/6] Generating Multimodal Feature Vectors (SigLIP / BGE / CLAP)...")
    img_embs, txt_embs, aud_embs = preprocessor.generate_all_embeddings()
    print("[Success] Preprocessing complete! Embeddings generated:")
    print(f"   Visual (SigLIP):   {img_embs.shape}")
    print(f"   Semantic (BGE):    {txt_embs.shape}")
    print(f"   Acoustic (CLAP):   {aud_embs.shape}")


def run_index(args):
    """Indexes preprocessed embeddings and frame metadata into Qdrant."""
    from src.embedder import MultimodalEmbedder

    media_folder = getattr(args, "media_folder", MEDIA_FOLDER_PATH)
    qdrant_path = getattr(args, "qdrant_path", QDRANT_DB_DIR)

    print(f"\n[Info] Initializing Multimodal Embedder (Storage: {qdrant_path})...")
    embedder = MultimodalEmbedder(
        media_folder_path=media_folder,
        qdrant_path=qdrant_path,
    )
    print("Generating embeddings and pushing named vectors into Qdrant...")
    embedder.run_pipeline(save_to_qdrant=True)
    print("[Success] Qdrant indexing completed successfully!")


def run_search(args):
    """Executes a multimodal query across the specified search mode."""
    from src.search_engine import MultimodalSearchEngine

    media_folder = getattr(args, "media_folder", MEDIA_FOLDER_PATH)
    qdrant_path = getattr(args, "qdrant_path", QDRANT_DB_DIR)
    device = getattr(args, "device", None) or get_default_device()

    engine = MultimodalSearchEngine(
        media_folder_path=media_folder,
        qdrant_path=qdrant_path,
        device=device
    )

    try:
        print(f"\nQuery: '{args.query}' | Mode: '{args.mode}' | Device: '{device}'")
        results = engine.search(
            query_text=args.query,
            mode=args.mode,
            character_name=getattr(args, "character", None),
            top_k=getattr(args, "top_k", DEFAULT_TOP_K),
            min_rrf_score=getattr(args, "min_rrf_score", MIN_RRF_SCORE),
            vis_weight=getattr(args, "vis_weight", None),
            sem_weight=getattr(args, "sem_weight", None),
            aud_weight=getattr(args, "aud_weight", None),
        )

        if not results:
            print("[Warning] No matching frames passed the confidence noise floor.")
            return

        print(f"\nTop {len(results)} Retrieved Frames:")
        print("-" * 80)
        for idx, r in enumerate(results, 1):
            char_info = f" | Characters: {r.get('characters')}" if r.get('characters') else ""
            print(f"#{idx:<2} Frame {r['frame_index']:<4} | Time: {r['timestamp']}s | Score: {r['score']:.4f}{char_info}")
            if r.get("transcript"):
                print(f"    Transcript: \"{r['transcript']}\"")
            path = r.get("file_path") or r.get("frame_path")
            if path:
                print(f"    Path: {path}")
        print("-" * 80)
    finally:
        engine.close()


def run_evaluate(args):
    """Runs retrieval benchmark evaluation against the golden dataset."""
    from src.search_engine import MultimodalSearchEngine
    from src.evaluate import evaluate_retrieval, run_benchmark_comparison

    media_folder = getattr(args, "media_folder", MEDIA_FOLDER_PATH)
    qdrant_path = getattr(args, "qdrant_path", QDRANT_DB_DIR)
    mode = getattr(args, "mode", "all")
    verbose = getattr(args, "verbose", False)

    dataset_path = os.path.join(media_folder, GOLDEN_DATASET_FILE)
    if not os.path.exists(dataset_path):
        print(f"[Error] Golden dataset file not found at: {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        golden_dataset = json.load(f)

    device = getattr(args, "device", None) or get_default_device()
    engine = MultimodalSearchEngine(
        media_folder_path=media_folder,
        qdrant_path=qdrant_path,
        device=device
    )

    try:
        if mode == "all":
            run_benchmark_comparison(
                search_engine=engine,
                dataset=golden_dataset,
                print_more_info=verbose,
            )
        else:
            res = evaluate_retrieval(
                search_engine=engine,
                dataset=golden_dataset,
                mode=mode,
                verbose=verbose,
            )
            print("\nEvaluation Results:")
            print(json.dumps({k: v for k, v in res.items() if k != "detailed_queries"}, indent=2))
    finally:
        engine.close()


def run_ui(args):
    """Launches the interactive Gradio web application."""
    from src.search_engine import MultimodalSearchEngine
    from src.gradio_app import build_gradio_ui

    media_folder = getattr(args, "media_folder", MEDIA_FOLDER_PATH)
    qdrant_path = getattr(args, "qdrant_path", QDRANT_DB_DIR)
    port = getattr(args, "port", DEFAULT_GRADIO_PORT)
    share = getattr(args, "share", False)
    device = getattr(args, "device", None) or get_default_device()

    engine = MultimodalSearchEngine(
        media_folder_path=media_folder,
        qdrant_path=qdrant_path,
        device=device
    )

    demo = build_gradio_ui(engine)
    print(f"\nLaunching Gradio UI on port {port} (share={share})...")
    demo.launch(server_port=port, share=share)


def main():
    parser = argparse.ArgumentParser(
        description="Unmayaverse: Multimodal AI Video Search & Retrieval Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Global options
    parser.add_argument("--media-folder", type=str, default=MEDIA_FOLDER_PATH, help="Media and assets folder")
    parser.add_argument("--qdrant-path", type=str, default=QDRANT_DB_DIR, help="Qdrant local storage path")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'mps', 'cpu')")

    # Top-level action flags (e.g., python -m src.main --evaluate --mode all)
    parser.add_argument("--preprocess", action="store_true", help="Extract raw data and generate embeddings")
    parser.add_argument("--embed", "--index", dest="embed", action="store_true", help="Index embeddings into Qdrant vector database")
    parser.add_argument("--evaluate", action="store_true", help="Run benchmark evaluation on golden dataset")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Gradio web application")

    # Shared parameters
    parser.add_argument("--mode", type=str, choices=SEARCH_MODES + ["all"], default="all", help="Search mode")
    parser.add_argument("--verbose", action="store_true", help="Print per-query evaluation details and modality breakdown")
    parser.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link")
    parser.add_argument("--min-rrf-score", type=float, default=MIN_RRF_SCORE, help="Minimum RRF score threshold")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline subcommands")

    # 1. Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Extract raw data and generate embeddings")
    p_prep.add_argument("--video-path", type=str, default=None, help="Path to input video file")
    p_prep.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame extraction rate")
    p_prep.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL, help="Whisper model size")
    p_prep.add_argument("--reference-frame", type=int, default=DEFAULT_REFERENCE_FRAME, help="Frame index containing Maya's reference face")
    p_prep.add_argument("--face-threshold", type=float, default=DEFAULT_FACE_THRESHOLD, help="Cosine similarity threshold for face matching")

    # 2. Index / Embed
    p_idx = subparsers.add_parser("index", help="Index embeddings into Qdrant vector database")
    subparsers.add_parser("embed", help="Index embeddings into Qdrant vector database")

    # 3. Search
    p_srch = subparsers.add_parser("search", help="Execute a multimodal retrieval query")
    p_srch.add_argument("query", type=str, help="Search query text")
    p_srch.add_argument("--mode", type=str, choices=SEARCH_MODES, default=SEARCH_MODE_VISUAL_TEXT_AUDIO, help="Search mode")
    p_srch.add_argument("--character", type=str, default=None, help="Filter by character identity")
    p_srch.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    p_srch.add_argument("--min-rrf-score", type=float, default=MIN_RRF_SCORE, help="Minimum RRF score threshold")
    p_srch.add_argument("--vis-weight", type=float, default=None, help="Visual modality weight")
    p_srch.add_argument("--sem-weight", type=float, default=None, help="Semantic dialogue weight")
    p_srch.add_argument("--aud-weight", type=float, default=None, help="Acoustic score weight")

    # 4. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation on golden dataset")
    p_eval.add_argument("--mode", type=str, choices=SEARCH_MODES + ["all"], default="all", help="Search mode to evaluate")
    p_eval.add_argument("--verbose", action="store_true", help="Print per-query evaluation details and modality breakdown")

    # 5. UI
    p_ui = subparsers.add_parser("ui", help="Launch interactive Gradio web application")
    p_ui.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Gradio server port")
    p_ui.add_argument("--share", action="store_true", help="Generate public Gradio share link")

    # 6. Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full pipeline: preprocess, index, then launch UI")
    p_pipe.add_argument("--video-path", type=str, default=None, help="Path to input video file")
    p_pipe.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Gradio server port")

    args = parser.parse_args()

    # Route top-level flags or subcommands
    if args.evaluate or args.command == "evaluate":
        run_evaluate(args)
    elif args.preprocess or args.command == "preprocess":
        run_preprocess(args)
    elif args.embed or args.command in ["index", "embed"]:
        run_index(args)
    elif args.ui or args.command == "ui":
        run_ui(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "pipeline":
        run_preprocess(args)
        run_index(args)
        run_ui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
