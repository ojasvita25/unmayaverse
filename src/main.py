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
    DEFAULT_MIN_Z_SCORE,
    DEFAULT_VIS_WEIGHT,
    DEFAULT_SEM_WEIGHT,
    DEFAULT_AUD_WEIGHT,
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

    print(f"\n[1/6] Initializing Preprocessor for: {args.media_folder}")
    preprocessor = DataPreprocessor(media_folder_path=args.media_folder)

    video_path = args.video_path or os.path.join(args.media_folder, VIDEO_FILE)
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found at: {video_path}")
        return

    print(f"Processing video: {video_path}")
    print("\n[2/6] Extracting 1 FPS Frames...")
    preprocessor.extract_frames(video_file=video_path, fps=args.fps)

    print("\n[3/6] Extracting Audio & Separating Vocals/Score (Demucs)...")
    preprocessor.extract_master_audio(video_file=video_path)
    preprocessor.separate_audio_demucs()

    print("\n[4/6] Transcribing Speech Dialogue (Whisper)...")
    preprocessor.transcribe_audio_whisper(model_name=args.whisper_model)

    print("\n[5/6] Extracting Character Face Embeddings (InsightFace)...")
    preprocessor.extract_maya_reference_embedding(reference_input=args.reference_frame)
    preprocessor.run_face_analysis(similarity_threshold=args.face_threshold)
    preprocessor.extract_and_align_all_raw_data()

    print("\n[6/6] Generating Multimodal Feature Vectors (SigLIP / BGE / CLAP)...")
    img_embs, txt_embs, aud_embs = preprocessor.generate_all_embeddings()
    print("[Success] Preprocessing complete! Embeddings generated:")
    print(f"   Visual (SigLIP):   {img_embs.shape}")
    print(f"   Semantic (BGE):    {txt_embs.shape}")
    print(f"   Acoustic (CLAP):   {aud_embs.shape}")


def run_index(args):
    """Indexes preprocessed embeddings and frame metadata into Qdrant."""
    from src.search_engine import MultimodalSearchEngine

    print(f"\n[Info] Initializing Qdrant Engine (Storage: {args.qdrant_path})...")
    engine = MultimodalSearchEngine(
        media_folder_path=args.media_folder,
        qdrant_path=args.qdrant_path,
        device=args.device or get_default_device()
    )
    print("Pushing named vectors (visual, semantic, audio, face) into Qdrant...")
    engine.index_embeddings_to_qdrant()
    print("[Success] Qdrant indexing completed successfully!")


def run_search(args):
    """Executes a multimodal query across the specified search mode."""
    from src.search_engine import MultimodalSearchEngine

    device = args.device or get_default_device()
    engine = MultimodalSearchEngine(
        media_folder_path=args.media_folder,
        qdrant_path=args.qdrant_path,
        device=device
    )

    print(f"\nQuery: '{args.query}' | Mode: '{args.mode}' | Device: '{device}'")
    results = engine.search(
        query_text=args.query,
        mode=args.mode,
        character_name=args.character,
        top_k=args.top_k,
        min_z_score=args.min_z_score,
        vis_weight=args.vis_weight,
        sem_weight=args.sem_weight,
        aud_weight=args.aud_weight
    )

    if not results:
        print("[Warning] No matching frames passed the confidence / Z-score noise floor.")
        return

    print(f"\nTop {len(results)} Retrieved Frames:")
    print("-" * 80)
    for idx, r in enumerate(results, 1):
        char_info = f" | Characters: {r.get('characters')}" if r.get('characters') else ""
        print(f"#{idx:<2} Frame {r['frame_index']:<4} | Time: {r['timestamp']}s | Score: {r['score']:.4f}{char_info}")
        if r.get("transcript"):
            print(f"    Transcript: \"{r['transcript']}\"")
        if r.get("frame_path"):
            print(f"    Path: {r['frame_path']}")
    print("-" * 80)


def run_evaluate(args):
    """Runs retrieval benchmark evaluation against the golden dataset."""
    from src.search_engine import MultimodalSearchEngine
    from src.evaluate import evaluate_retrieval

    dataset_path = os.path.join(args.media_folder, GOLDEN_DATASET_FILE)
    if not os.path.exists(dataset_path):
        print(f"[Error] Golden dataset file not found at: {dataset_path}")
        return

    with open(dataset_path, "r") as f:
        golden_dataset = json.load(f)

    device = args.device or get_default_device()
    engine = MultimodalSearchEngine(
        media_folder_path=args.media_folder,
        qdrant_path=args.qdrant_path,
        device=device
    )

    modes_to_eval = SEARCH_MODES if args.mode == "all" else [args.mode]
    summary_results = []

    for mode in modes_to_eval:
        res = evaluate_retrieval(
            search_engine=engine,
            dataset=golden_dataset,
            mode=mode,
            min_z_score=args.min_z_score
        )
        summary_results.append(res)

    print("\n" + "=" * 80)
    print(" SUMMARY BENCHMARK COMPARISON ACROSS MODES")
    print("=" * 80)
    for res in summary_results:
        print(
            f"Mode: {res['mode']:<22} | MRR: {res['mrr']:.3f} | "
            f"Recall@1: {res['recall_at_k'].get(1, 0.0):.3f} | "
            f"Recall@5: {res['recall_at_k'].get(5, 0.0):.3f} | "
            f"Rejection Acc: {res['rejection_accuracy_pct']:.1f}%"
        )


def run_ui(args):
    """Launches the interactive Gradio web application."""
    from src.search_engine import MultimodalSearchEngine
    from src.gradio_app import build_gradio_ui

    device = args.device or get_default_device()
    engine = MultimodalSearchEngine(
        media_folder_path=args.media_folder,
        qdrant_path=args.qdrant_path,
        device=device
    )

    demo = build_gradio_ui(engine, media_folder_path=args.media_folder)
    print(f"\nLaunching Gradio UI on port {args.port} (share={args.share})...")
    demo.launch(server_port=args.port, share=args.share)


def main():
    parser = argparse.ArgumentParser(
        description="Unmayaverse: Multimodal AI Video Search & Retrieval Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--media-folder", type=str, default=MEDIA_FOLDER_PATH, help="Media and assets folder")
    parser.add_argument("--qdrant-path", type=str, default=QDRANT_DB_DIR, help="Qdrant local storage path")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cuda', 'mps', 'cpu')")

    subparsers = parser.add_subparsers(dest="command", help="Pipeline command to execute")

    # 1. Preprocess
    p_prep = subparsers.add_parser("preprocess", help="Extract raw data and generate embeddings")
    p_prep.add_argument("--video-path", type=str, default=None, help="Path to input video file")
    p_prep.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frame extraction rate")
    p_prep.add_argument("--whisper-model", type=str, default=DEFAULT_WHISPER_MODEL, help="Whisper model size")
    p_prep.add_argument("--reference-frame", type=int, default=DEFAULT_REFERENCE_FRAME, help="Frame index containing Maya's reference face")
    p_prep.add_argument("--face-threshold", type=float, default=DEFAULT_FACE_THRESHOLD, help="Cosine similarity threshold for face matching")

    # 2. Index
    p_idx = subparsers.add_parser("index", help="Index embeddings into Qdrant vector database")

    # 3. Search
    p_srch = subparsers.add_parser("search", help="Execute a multimodal retrieval query")
    p_srch.add_argument("query", type=str, help="Search query text")
    p_srch.add_argument("--mode", type=str, choices=SEARCH_MODES, default=SEARCH_MODE_VISUAL_TEXT_AUDIO, help="Search mode")
    p_srch.add_argument("--character", type=str, default=None, help="Filter by character identity")
    p_srch.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results to return")
    p_srch.add_argument("--min-z-score", type=float, default=DEFAULT_MIN_Z_SCORE, help="Z-score threshold for outlier filtering")
    p_srch.add_argument("--vis-weight", type=float, default=DEFAULT_VIS_WEIGHT, help="Visual modality weight")
    p_srch.add_argument("--sem-weight", type=float, default=DEFAULT_SEM_WEIGHT, help="Semantic dialogue weight")
    p_srch.add_argument("--aud-weight", type=float, default=DEFAULT_AUD_WEIGHT, help="Acoustic score weight")

    # 4. Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run benchmark evaluation on golden dataset")
    p_eval.add_argument("--mode", type=str, choices=SEARCH_MODES + ["all"], default="all", help="Search mode to evaluate")
    p_eval.add_argument("--min-z-score", type=float, default=DEFAULT_MIN_Z_SCORE, help="Z-score threshold")

    # 5. UI
    p_ui = subparsers.add_parser("ui", help="Launch interactive Gradio web application")
    p_ui.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Gradio server port")
    p_ui.add_argument("--share", action="store_true", help="Generate public Gradio share link")

    # 6. Pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run full pipeline: preprocess, index, then launch UI")
    p_pipe.add_argument("--video-path", type=str, default=None, help="Path to input video file")
    p_pipe.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Gradio server port")

    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocess(args)
    elif args.command == "index":
        run_index(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "ui":
        run_ui(args)
    elif args.command == "pipeline":
        run_preprocess(args)
        run_index(args)
        run_ui(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
