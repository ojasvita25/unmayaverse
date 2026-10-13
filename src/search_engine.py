"""
Unmayaverse: Multimodal Search Engine
Queries local Qdrant Vector DB across Visual, Dialogue, and Acoustic score modes with RRF fusion.
"""

import os
import json
import numpy as np
import torch
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .config import (
    PROJECT_ROOT,
    MEDIA_FOLDER_PATH,
    QDRANT_DB_DIR,
    QDRANT_COLLECTION_NAME,
    FRAME_METADATA_JSON,
    VISUAL_MODEL_NAME,
    SEMANTIC_MODEL_NAME,
    ACOUSTIC_MODEL_NAME,
    RRF_K_CONSTANT,
    VISUAL_NOISE_FLOOR,
    SEMANTIC_NOISE_FLOOR,
    ACOUSTIC_NOISE_FLOOR,
    SEARCH_MODE_VISUAL,
    SEARCH_MODE_VISUAL_TEXT,
    SEARCH_MODE_VISUAL_TEXT_AUDIO,
    DEFAULT_WEIGHTS_VIS_TEXT,
    DEFAULT_WEIGHTS_TRI_MODAL,
    MIN_RRF_SCORE,

)


class MultimodalSearchEngine:
    """
    Multimodal AI Search Engine powered by local Qdrant Vector Database.
    Supports 3 Search Modes + Automatic Character Filtering:
    1. 'visual' (SigLIP)
    2. 'visual+text' (SigLIP visual + BGE dialogue text RRF)
    3. 'visual+text+audio' (SigLIP visual + BGE dialogue + CLAP audio RRF)
    """

    def __init__(self, media_folder_path: str = MEDIA_FOLDER_PATH, qdrant_path: Optional[str] = None, device: Optional[str] = None):
        if not os.path.isabs(media_folder_path) and not os.path.exists(media_folder_path):
            candidate = os.path.join(PROJECT_ROOT, media_folder_path)
            if os.path.exists(candidate):
                media_folder_path = candidate

        self.media_folder = media_folder_path
        self.qdrant_path = qdrant_path if qdrant_path else os.path.join(self.media_folder, "qdrant_db")
        self.device = device or self._detect_device()

        # Connect to local embedded Qdrant
        self.client = QdrantClient(path=self.qdrant_path)

        # Lazy model encoders
        self.visual_model = None
        self.semantic_model = None
        self.clap_model = None
        self.clap_proc = None

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

    def load_encoders(self, load_vis: bool = True, load_sem: bool = True, load_aud: bool = True):
        """Loads model encoders lazily only when needed."""
        from sentence_transformers import SentenceTransformer

        if load_vis and self.visual_model is None:
            print(f"Loading Visual Encoder ({VISUAL_MODEL_NAME})...")
            self.visual_model = SentenceTransformer(VISUAL_MODEL_NAME, device=self.device)

        if load_sem and self.semantic_model is None:
            print(f"Loading Semantic Encoder ({SEMANTIC_MODEL_NAME})...")
            self.semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device=self.device)

        if load_aud and self.clap_model is None:
            import transformers
            transformers.logging.set_verbosity_error()
            from transformers import ClapModel, ClapProcessor

            print(f"Loading Acoustic Encoder ({ACOUSTIC_MODEL_NAME})...")
            self.clap_proc = ClapProcessor.from_pretrained(ACOUSTIC_MODEL_NAME)
            self.clap_model = ClapModel.from_pretrained(ACOUSTIC_MODEL_NAME).to(self.device)

    @staticmethod
    def compute_z_scores(scores: np.ndarray) -> np.ndarray:
        std = np.std(scores)
        if std < 1e-6:
            return np.zeros_like(scores)
        return (scores - np.mean(scores)) / std

    def get_available_characters(self) -> List[str]:
        """Returns character identities tagged in metadata."""
        return ["All Characters (Auto-Detect)", "Maya"]

    def search(
        self,
        query_text: str,
        mode: str = SEARCH_MODE_VISUAL_TEXT_AUDIO,
        character_name: Optional[str] = None,
        top_k: int = 5,
        min_rrf_score: Optional[float] = None,
        vis_weight: Optional[float] = None,
        sem_weight: Optional[float] = None,
        aud_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Executes multimodal search across Qdrant with optional character filtering and RRF fusion."""
        if not self.client.collection_exists(QDRANT_COLLECTION_NAME):
            print(f"[Error] Qdrant collection '{QDRANT_COLLECTION_NAME}' does not exist.")
            return []

        coll_info = self.client.get_collection(QDRANT_COLLECTION_NAME)
        total_points = coll_info.points_count
        if total_points == 0:
            print("[Warning] No points indexed in Qdrant.")
            return []

        load_vis = mode in [SEARCH_MODE_VISUAL, SEARCH_MODE_VISUAL_TEXT, SEARCH_MODE_VISUAL_TEXT_AUDIO]
        load_sem = mode in [SEARCH_MODE_VISUAL_TEXT, SEARCH_MODE_VISUAL_TEXT_AUDIO]
        load_aud = mode == SEARCH_MODE_VISUAL_TEXT_AUDIO
        self.load_encoders(load_vis, load_sem, load_aud)

        # Automatic character identity detection from query text or explicit filter
        target_char = character_name
        if target_char in [None, "All Characters", "All Characters (Auto-Detect)"]:
            # Auto-detect known character keywords in query
            q_lower = query_text.lower()
            if "maya" in q_lower:
                target_char = "Maya"
            else:
                target_char = None

        query_filter = None
        if target_char:
            query_filter = Filter(
                must=[FieldCondition(key="characters", match=MatchValue(value=target_char))]
            )

        # -------------------------------------------------------------
        # DYNAMIC WEIGHT RESOLUTION & NORMALIZATION 
        # -------------------------------------------------------------
        if mode == SEARCH_MODE_VISUAL_TEXT:
            vw = vis_weight if vis_weight is not None else DEFAULT_WEIGHTS_VIS_TEXT["visual"]
            sw = sem_weight if sem_weight is not None else DEFAULT_WEIGHTS_VIS_TEXT["semantic"]
            total = vw + sw
            vis_w, sem_w, aud_w = vw / total, sw / total, 0.0
        elif mode == SEARCH_MODE_VISUAL_TEXT_AUDIO:
            vw = vis_weight if vis_weight is not None else DEFAULT_WEIGHTS_TRI_MODAL["visual"]
            sw = sem_weight if sem_weight is not None else DEFAULT_WEIGHTS_TRI_MODAL["semantic"]
            aw = aud_weight if aud_weight is not None else DEFAULT_WEIGHTS_TRI_MODAL["audio"]
            total = vw + sw + aw
            vis_w, sem_w, aud_w = vw / total, sw / total, aw / total
        else:  # Visual only
            vis_w, sem_w, aud_w = 1.0, 0.0, 0.0

        # -----------------------------------------------------------------
        # MODE 1: VISUAL ONLY
        # -----------------------------------------------------------------
        if mode == SEARCH_MODE_VISUAL:
            v_query = self.visual_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
            hits = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                using="visual",
                query=v_query.tolist(),
                query_filter=query_filter,
                limit=total_points
            ).points

            if not hits:
                return []

            scores = np.array([pt.score for pt in hits])
            # z_scores = self.compute_z_scores(scores)
            results = []

            for rank_idx, pt in enumerate(hits):
                score = float(pt.score)
                # z_val = float(z_scores[rank_idx])
                if score >= VISUAL_NOISE_FLOOR:
                    results.append({
                        "frame_index": pt.payload["frame_index"],
                        "timestamp": pt.payload["timestamp"],
                        "score": score,
                        "file_path": pt.payload["file_path"],
                        "transcript": pt.payload.get("transcript", ""),
                        "characters": pt.payload.get("characters", []),
                        "mode": mode
                    })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        
        

        # -----------------------------------------------------------------
        # MODE 2 & 3: RECIPROCAL RANK FUSION (RRF) - COSINE THRESHOLD ONLY
        # -----------------------------------------------------------------
        rrf_scores: Dict[int, float] = {}
        payload_map: Dict[int, Any] = {}

        # 1. Visual Query (SigLIP)
        v_query = self.visual_model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        v_hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            using="visual",
            query=v_query.tolist(),
            query_filter=query_filter,
            limit=total_points
        ).points

        if v_hits:
            valid_vis_rank = 0  # Reindexing counter for surviving visual hits
            for pt in v_hits:
                f_idx = pt.payload["frame_index"]
                payload_map[f_idx] = pt.payload
                # Pure cosine thresholding gate
                if pt.score >= VISUAL_NOISE_FLOOR:
                    # print('siglip cosine',pt.score)
                    rrf_scores[f_idx] = rrf_scores.get(f_idx, 0.0) + vis_w * (1.0 / (RRF_K_CONSTANT + valid_vis_rank + 1))
                    valid_vis_rank += 1

        # 2. Semantic Query (BGE)
        sem_prompt = f"Represent this sentence for searching relevant passages: {query_text}"
        s_query = self.semantic_model.encode([sem_prompt], convert_to_numpy=True, normalize_embeddings=True)[0]
        s_hits = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            using="semantic",
            query=s_query.tolist(),
            query_filter=query_filter,
            limit=total_points
        ).points

        if s_hits:
            valid_sem_rank = 0  # Reindexing counter for surviving semantic hits
            for pt in s_hits:
                f_idx = pt.payload["frame_index"]
                payload_map[f_idx] = pt.payload
                # Pure cosine thresholding gate
                if pt.score >= SEMANTIC_NOISE_FLOOR:
                    # print('bge cosine',pt.score)
                    rrf_scores[f_idx] = rrf_scores.get(f_idx, 0.0) + sem_w * (1.0 / (RRF_K_CONSTANT + valid_sem_rank + 1))
                    valid_sem_rank += 1

        # 3. Acoustic Query (CLAP) (Mode 3 Only)
        if mode == SEARCH_MODE_VISUAL_TEXT_AUDIO and self.clap_model is not None:
            clap_inputs = self.clap_proc(text=[query_text], return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.clap_model.get_text_features(**clap_inputs)
                a_query = getattr(out, "pooler_output", out)
                if isinstance(a_query, torch.Tensor):
                    a_query = a_query.squeeze().cpu().numpy()
                a_query = a_query / (np.linalg.norm(a_query) + 1e-8)

            a_hits = self.client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                using="audio",
                query=a_query.tolist(),
                query_filter=query_filter,
                limit=total_points
            ).points

            if a_hits:
                valid_aud_rank = 0  # Reindexing counter for surviving audio hits
                for pt in a_hits:
                    f_idx = pt.payload["frame_index"]
                    payload_map[f_idx] = pt.payload
                    # Pure cosine thresholding gate
                    if pt.score >= ACOUSTIC_NOISE_FLOOR:
                        # print('audio clap cosine',pt.score)
                        rrf_scores[f_idx] = rrf_scores.get(f_idx, 0.0) + aud_w * (1.0 / (RRF_K_CONSTANT + valid_aud_rank + 1))
                        valid_aud_rank += 1

        # Final Aggregation & Processing (Gated by MIN_RRF_SCORE)
        rrf_threshold = min_rrf_score if min_rrf_score is not None else MIN_RRF_SCORE
        results = []
        for f_idx, score in rrf_scores.items():
            # Discard any candidate that fails the fused confidence threshold
            if score < rrf_threshold:
                continue

            meta = payload_map[f_idx]
            results.append({
                "frame_index": f_idx,
                "timestamp": meta.get("timestamp", f_idx),
                "score": float(score),
                "file_path": meta.get("file_path", ""),
                "transcript": meta.get("transcript", ""),
                "characters": meta.get("characters", []),
                "mode": mode
            })


        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]



    def close(self):
        """Releases local Qdrant database file lock."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()

