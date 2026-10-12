"""
Unmayaverse: Multimodal Embedding Generation & Direct Vector Indexing
Loads SigLIP, BGE, and CLAP models to generate feature vectors and saves directly into Qdrant.
"""

import os
import time
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .config import (
    PROJECT_ROOT,
    MEDIA_FOLDER_PATH,
    MASTER_AUDIO_FILE,
    FRAME_METADATA_JSON,
    IMAGE_EMBEDDINGS_NPY,
    TEXT_EMBEDDINGS_NPY,
    AUDIO_EMBEDDINGS_NPY,
    QDRANT_DB_DIR,
    QDRANT_COLLECTION_NAME,
    SIGLIP_VECTOR_DIM,
    BGE_VECTOR_DIM,
    CLAP_VECTOR_DIM,
    VISUAL_MODEL_NAME,
    SEMANTIC_MODEL_NAME,
    ACOUSTIC_MODEL_NAME,
)


class MultimodalEmbedder:
    """
    Generates multimodal feature representations and persists them directly into Qdrant:
    - Visual: SigLIP SO400M (1152-dim)
    - Semantic: BGE Large v1.5 (1024-dim)
    - Acoustic: CLAP HTSAT (512-dim)
    """

    def __init__(self, media_folder_path: str = MEDIA_FOLDER_PATH, qdrant_path: Optional[str] = None):
        # Auto-resolve relative paths if called from inside notebooks/ or another subfolder
        if not os.path.isabs(media_folder_path) and not os.path.exists(media_folder_path):
            candidate = os.path.join(PROJECT_ROOT, media_folder_path)
            if os.path.exists(candidate):
                media_folder_path = candidate

        self.media_folder = media_folder_path
        self.qdrant_path = qdrant_path if qdrant_path else os.path.join(self.media_folder, "qdrant_db")
        self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def embed_visual(self, frame_metadata: list) -> np.ndarray:
        """Generates SigLIP image embeddings for all video frames with progress bar."""
        from sentence_transformers import SentenceTransformer

        start_time = time.time()
        print(f"\n[1/4] Visual Encoding: Loading SigLIP ({VISUAL_MODEL_NAME}) on {self.device}...")
        vis_model = SentenceTransformer(VISUAL_MODEL_NAME, device=self.device)

        images = [Image.open(f["file_path"]).convert("RGB") for f in frame_metadata]
        img_embs = vis_model.encode(
            images,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        img_embs = np.ascontiguousarray(img_embs, dtype=np.float32)

        out_path = os.path.join(self.media_folder, IMAGE_EMBEDDINGS_NPY)
        np.save(out_path, img_embs)
        elapsed = time.time() - start_time
        fps = len(frame_metadata) / max(elapsed, 0.001)
        print(f"[Success] Visual SigLIP: {img_embs.shape} generated in {elapsed:.1f}s ({fps:.1f} frames/s)")
        return img_embs

    def embed_semantic(self, frame_metadata: list) -> np.ndarray:
        """Generates BGE semantic embeddings for dialogue transcripts with progress bar."""
        from sentence_transformers import SentenceTransformer

        start_time = time.time()
        print(f"\n[2/4] Semantic Encoding: Loading BGE ({SEMANTIC_MODEL_NAME}) on {self.device}...")
        sem_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=self.device)

        texts = [f["text"] for f in frame_metadata]
        txt_embs = sem_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        txt_embs = np.ascontiguousarray(txt_embs, dtype=np.float32)

        out_path = os.path.join(self.media_folder, TEXT_EMBEDDINGS_NPY)
        np.save(out_path, txt_embs)
        elapsed = time.time() - start_time
        fps = len(frame_metadata) / max(elapsed, 0.001)
        print(f"[Success] Semantic BGE: {txt_embs.shape} generated in {elapsed:.1f}s ({fps:.1f} frames/s)")
        return txt_embs

    def embed_acoustic(self, num_frames: int) -> np.ndarray:
        """Generates CLAP acoustic embeddings for background score/audio with progress bar."""
        import librosa
        import torch
        import transformers
        transformers.logging.set_verbosity_error()
        from transformers import ClapModel, ClapProcessor

        start_time = time.time()
        print(f"\n[3/4] Acoustic Encoding: Loading CLAP ({ACOUSTIC_MODEL_NAME}) on {self.device}...")
        clap_proc = ClapProcessor.from_pretrained(ACOUSTIC_MODEL_NAME)
        clap_model = ClapModel.from_pretrained(ACOUSTIC_MODEL_NAME).to(self.device)

        audio_path = os.path.join(self.media_folder, "htdemucs", os.path.splitext(MASTER_AUDIO_FILE)[0], "no_vocals.wav")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.media_folder, MASTER_AUDIO_FILE)

        if not os.path.exists(audio_path):
            print("[Warning] No audio found for CLAP encoding. Defaulting to zeros.")
            aud_embs = np.zeros((num_frames, CLAP_VECTOR_DIM), dtype=np.float32)
            np.save(os.path.join(self.media_folder, AUDIO_EMBEDDINGS_NPY), aud_embs)
            return aud_embs

        audio_data, sr = librosa.load(audio_path, sr=48000)
        audio_embs = []

        with tqdm(total=num_frames, desc="Encoding CLAP audio chunks", unit="frame") as pbar:
            for idx in range(num_frames):
                start_sample = int(idx * sr)
                end_sample = int((idx + 1) * sr)
                chunk = audio_data[start_sample:end_sample]
                if len(chunk) == 0:
                    chunk = np.zeros(sr, dtype=np.float32)

                inputs = clap_proc(audio=chunk, sampling_rate=sr, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = clap_model.get_audio_features(**inputs)
                    emb = getattr(out, "pooler_output", out)
                    if isinstance(emb, torch.Tensor):
                        emb = emb.squeeze().cpu().numpy()
                    emb = emb / (np.linalg.norm(emb) + 1e-8)
                audio_embs.append(emb)

                pbar.update(1)

        aud_embs = np.ascontiguousarray(np.vstack(audio_embs), dtype=np.float32)
        out_path = os.path.join(self.media_folder, AUDIO_EMBEDDINGS_NPY)
        np.save(out_path, aud_embs)
        elapsed = time.time() - start_time
        fps = num_frames / max(elapsed, 0.001)
        print(f"[Success] Acoustic CLAP: {aud_embs.shape} generated in {elapsed:.1f}s ({fps:.1f} frames/s)")
        return aud_embs

    def save_to_qdrant(self, img_embs: np.ndarray, txt_embs: np.ndarray, aud_embs: np.ndarray, frame_metadata: list) -> str:
        """Saves embeddings and frame metadata directly into embedded Qdrant DB with progress bar."""
        start_time = time.time()
        print(f"\n[4/4] Qdrant Indexing: Initializing local vector database at: {self.qdrant_path}...")
        os.makedirs(self.qdrant_path, exist_ok=True)
        client = QdrantClient(path=self.qdrant_path)

        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            print(f"Creating multi-vector Qdrant collection: '{QDRANT_COLLECTION_NAME}'...")
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config={
                    "visual": VectorParams(size=SIGLIP_VECTOR_DIM, distance=Distance.COSINE),
                    "semantic": VectorParams(size=BGE_VECTOR_DIM, distance=Distance.COSINE),
                    "audio": VectorParams(size=CLAP_VECTOR_DIM, distance=Distance.COSINE),
                }
            )

        points = []
        for idx in tqdm(range(len(frame_metadata)), desc="Assembling Qdrant payloads", unit="point"):
            meta = frame_metadata[idx]
            v_vec = img_embs[idx].tolist() if img_embs is not None else [0.0] * SIGLIP_VECTOR_DIM
            s_vec = txt_embs[idx].tolist() if txt_embs is not None else [0.0] * BGE_VECTOR_DIM
            a_vec = aud_embs[idx].tolist() if aud_embs is not None else [0.0] * CLAP_VECTOR_DIM

            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "visual": v_vec,
                        "semantic": s_vec,
                        "audio": a_vec
                    },
                    payload={
                        "frame_index": idx,
                        "timestamp": meta.get("timestamp", idx),
                        "file_path": meta.get("file_path", ""),
                        "transcript": meta.get("text", ""),
                        "characters": meta.get("characters", [])
                    }
                )
            )

        print(f"Upserting {len(points)} multimodal points into Qdrant...")
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        elapsed = time.time() - start_time
        print(f"[Success] All points indexed in Qdrant in {elapsed:.1f}s!")
        return self.qdrant_path

    def run_pipeline(self, max_frames: Optional[int] = None, save_to_qdrant: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs embedding and indexing pipeline. If max_frames is provided, only processes the first N frames."""
        overall_start = time.time()

        frame_meta_path = os.path.join(self.media_folder, FRAME_METADATA_JSON)
        if not os.path.exists(frame_meta_path):
            raise FileNotFoundError(f"Frame metadata not found at: {frame_meta_path}. Run preprocessing first.")

        with open(frame_meta_path, "r") as f:
            frame_metadata = json.load(f)

        # Slice to test subset if specified
        if max_frames is not None:
            frame_metadata = frame_metadata[:max_frames]

        num_frames = len(frame_metadata)
        print("=" * 70)
        print(f"STARTING MULTIMODAL EMBEDDING PIPELINE FOR {num_frames} FRAMES")
        print("=" * 70)

        img_embs = self.embed_visual(frame_metadata)
        txt_embs = self.embed_semantic(frame_metadata)
        aud_embs = self.embed_acoustic(num_frames)

        if save_to_qdrant:
            self.save_to_qdrant(img_embs, txt_embs, aud_embs, frame_metadata)

        total_elapsed = time.time() - overall_start
        print("\n" + "=" * 70)
        print(f"[COMPLETE] Multimodal Pipeline Finished in {total_elapsed:.1f}s ({total_elapsed/60:.2f} mins)!")
        print("=" * 70)

        return img_embs, txt_embs, aud_embs

