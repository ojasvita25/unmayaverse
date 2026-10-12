"""
Unmayaverse: Multimodal Data Preprocessor & Feature Extraction Pipeline
Handles frame extraction, audio separation, transcription, face analysis, and multimodal embeddings.
"""

import os
import glob
import json
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

from .config import (
    MEDIA_FOLDER_PATH,
    VIDEO_FILE,
    MASTER_AUDIO_FILE,
    FRAMES_DIR_NAME,
    TRANSCRIPT_JSON,
    FRAME_METADATA_JSON,
    FACE_EMBEDDINGS_JSON,
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
    FACE_MODEL_NAME,
    DEFAULT_FPS,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_REFERENCE_FRAME,
    DEFAULT_FACE_THRESHOLD,
    FRAME_MAX_WIDTH,
    FRAME_JPEG_QUALITY,
)



class DataPreprocessor:
    """
    Handles end-to-end raw data extraction from video:
    1. Encoded video frame extraction (FFmpeg at 1 FPS, max 720p, optimized JPEG)
    2. Master audio extraction & Demucs vocal/instrumental separation
    3. Whisper speech dialogue transcription
    4. InsightFace character identity matching (Maya)
    5. Feature encoding with SigLIP, BGE, and CLAP
    """

    def __init__(self, media_folder_path: str = MEDIA_FOLDER_PATH):
        self.media_folder = media_folder_path
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

    def extract_frames(
        self,
        video_file: Optional[str] = None,
        fps: int = DEFAULT_FPS,
        max_width: int = FRAME_MAX_WIDTH,
        quality: int = FRAME_JPEG_QUALITY
    ) -> str:
        """Extracts lightweight encoded JPEG frames using FFmpeg."""
        target_video = video_file if video_file else os.path.join(self.media_folder, VIDEO_FILE)
        output_folder = os.path.join(self.media_folder, FRAMES_DIR_NAME)
        os.makedirs(output_folder, exist_ok=True)

        if not os.path.exists(target_video):
            print(f"[Error] Source video file not found at: {target_video}")
            return output_folder

        output_pattern = os.path.join(output_folder, "frame_%04d.jpg")
        vf_filter = f"fps={fps},scale='min({max_width},iw)':-2"
        cmd = [
            "ffmpeg", "-y", "-i", target_video,
            "-vf", vf_filter,
            "-q:v", str(quality),
            output_pattern
        ]
        print(f"Extracting encoded {max_width}p frames (q={quality}) from {target_video}...")
        subprocess.run(cmd, check=True)
        print(f"[Success] Extracted frames saved to: {output_folder}")
        return output_folder

    def extract_master_audio(self, video_file: Optional[str] = None) -> str:
        """Extracts 48kHz WAV audio from video using FFmpeg."""
        target_video = video_file if video_file else os.path.join(self.media_folder, VIDEO_FILE)
        audio_output = os.path.join(self.media_folder, MASTER_AUDIO_FILE)

        if not os.path.exists(target_video):
            print(f"[Error] Source video not found for audio extraction: {target_video}")
            return audio_output

        cmd = [
            "ffmpeg", "-y", "-i", target_video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "2",
            audio_output
        ]
        print(f"Extracting 48kHz WAV audio to {audio_output}...")
        subprocess.run(cmd, check=True)
        print(f"[Success] Audio extracted to: {audio_output}")
        return audio_output

    def separate_audio_demucs(self) -> str:
        """Separates vocals and background instrumental score using Demucs."""
        audio_path = os.path.join(self.media_folder, MASTER_AUDIO_FILE)
        if not os.path.exists(audio_path):
            audio_path = self.extract_master_audio()

        if not os.path.exists(audio_path):
            print(f"[Warning] Audio file not found for Demucs separation: {audio_path}")
            return ""

        print("Separating audio streams with Demucs (Vocals vs Score/FX)...")
        cmd = [
            "demucs", "--two-stems=vocals",
            "-o", self.media_folder,
            audio_path
        ]
        subprocess.run(cmd, check=True)
        separated_path = os.path.join(self.media_folder, "htdemucs", os.path.splitext(MASTER_AUDIO_FILE)[0], "no_vocals.wav")
        print(f"[Success] Instrumental score isolated at: {separated_path}")
        return separated_path

    def transcribe_audio_whisper(self, model_name: str = DEFAULT_WHISPER_MODEL) -> str:
        """Transcribes dialogue speech using Whisper and writes timestamp mapping."""
        import ssl
        import whisper

        # Bypass SSL certificate verification for downloading model weights
        ssl._create_default_https_context = ssl._create_unverified_context

        audio_path = os.path.join(self.media_folder, MASTER_AUDIO_FILE)
        output_json = os.path.join(self.media_folder, TRANSCRIPT_JSON)

        if not os.path.exists(audio_path):
            audio_path = self.extract_master_audio()

        if not os.path.exists(audio_path):
            print("[Warning] Audio file not found for Whisper transcription.")
            return output_json

        print(f"Transcribing audio with Whisper ({model_name})...")
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)

        fps_text_mapping = {}
        for segment in result.get("segments", []):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            for sec in range(int(start), int(end) + 1):
                fps_text_mapping[str(sec)] = text

        with open(output_json, "w") as f:
            json.dump(fps_text_mapping, f, indent=2)

        print(f"[Success] Saved transcript mapping to: {output_json}")
        return output_json

    def extract_maya_reference_embedding(self, reference_input: int = DEFAULT_REFERENCE_FRAME) -> np.ndarray:
        """Extracts 512-dim InsightFace face vector from Maya's reference frame."""
        import cv2
        from insightface.app import FaceAnalysis

        frames_folder = os.path.join(self.media_folder, FRAMES_DIR_NAME)
        frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

        if not frame_files or reference_input >= len(frame_files):
            raise ValueError(f"Frame index {reference_input} out of bounds ({len(frame_files)} frames found).")

        img_path = frame_files[reference_input]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load reference frame at: {img_path}")

        providers = ['CUDAExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        app = FaceAnalysis(name=FACE_MODEL_NAME, providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(img)

        if not faces:
            raise ValueError(f"No faces detected in reference frame: {img_path}")

        faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        primary_face = faces[0]
        emb = primary_face.embedding / (np.linalg.norm(primary_face.embedding) + 1e-8)

        x1, y1, x2, y2 = [int(v) for v in primary_face.bbox]
        h, w, _ = img.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = img[y1:y2, x1:x2]

        thumb_path = os.path.join(self.media_folder, "maya_reference_crop.jpg")
        cv2.imwrite(thumb_path, face_crop)

        ref_emb_path = os.path.join(self.media_folder, "maya_reference_embedding.npy")
        np.save(ref_emb_path, emb)

        print(f"[Success] Extracted Maya reference face vector from frame #{reference_input}")
        return emb

    def run_face_analysis(
        self,
        min_det_score: float = 0.50,
        min_face_size: int = 40,
        similarity_threshold: float = DEFAULT_FACE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """Matches Maya's face directly across all frames using cosine similarity."""
        import cv2
        from insightface.app import FaceAnalysis

        ref_emb_path = os.path.join(self.media_folder, "maya_reference_embedding.npy")
        if not os.path.exists(ref_emb_path):
            self.extract_maya_reference_embedding()

        maya_ref_emb = np.load(ref_emb_path)
        frames_folder = os.path.join(self.media_folder, FRAMES_DIR_NAME)
        frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

        if not frame_files:
            print("[Warning] No frames found for face analysis.")
            return []

        providers = ['CUDAExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
        app = FaceAnalysis(name=FACE_MODEL_NAME, providers=providers)
        app.prepare(ctx_id=0, det_size=(640, 640))

        face_data = []
        matched_maya_count = 0

        for idx, file_path in enumerate(tqdm(frame_files, desc="Matching Maya faces")):
            img = cv2.imread(file_path)
            if img is None:
                continue
            faces = app.get(img)

            for face in faces:
                if float(face.det_score) < min_det_score:
                    continue

                bbox = [int(x) for x in face.bbox]
                w, h = max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1])
                if w < min_face_size or h < min_face_size:
                    continue

                face_emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
                similarity = float(np.dot(maya_ref_emb, face_emb))
                is_maya = similarity >= similarity_threshold

                if is_maya:
                    matched_maya_count += 1

                face_data.append({
                    "frame_index": idx,
                    "file_path": file_path,
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "embedding": face.embedding.tolist(),
                    "similarity_to_maya": similarity,
                    "character": "Maya" if is_maya else "Unknown"
                })

        out_path = os.path.join(self.media_folder, FACE_EMBEDDINGS_JSON)
        with open(out_path, "w") as f:
            json.dump(face_data, f, indent=2)

        print(f"[Success] Detected {len(face_data)} faces across frames. Matched Maya in {matched_maya_count} instances.")
        return face_data

    def extract_and_align_all_raw_data(self) -> List[Dict[str, Any]]:
        """Aligns extracted frames, dialogue transcript, and character identities."""
        frames_folder = os.path.join(self.media_folder, FRAMES_DIR_NAME)
        frame_files = sorted(glob.glob(os.path.join(frames_folder, "*.jpg")))

        transcript_path = os.path.join(self.media_folder, TRANSCRIPT_JSON)
        fps_text_mapping = {}
        if os.path.exists(transcript_path):
            with open(transcript_path, "r") as f:
                fps_text_mapping = json.load(f)

        face_path = os.path.join(self.media_folder, FACE_EMBEDDINGS_JSON)
        frame_characters = {}
        if os.path.exists(face_path):
            with open(face_path, "r") as f:
                face_data = json.load(f)
            for face in face_data:
                if face.get("character") == "Maya":
                    f_idx = face["frame_index"]
                    if f_idx not in frame_characters:
                        frame_characters[f_idx] = []
                    if "Maya" not in frame_characters[f_idx]:
                        frame_characters[f_idx].append("Maya")

        aligned_items = []
        for idx, file_path in enumerate(frame_files):
            timestamp = idx
            text = fps_text_mapping.get(str(timestamp), "silence")
            chars = frame_characters.get(idx, [])

            aligned_items.append({
                "frame_index": idx,
                "timestamp": timestamp,
                "file_path": file_path,
                "text": text,
                "characters": chars
            })

        out_path = os.path.join(self.media_folder, FRAME_METADATA_JSON)
        with open(out_path, "w") as f:
            json.dump(aligned_items, f, indent=2)

        print(f"[Success] Aligned raw metadata for {len(aligned_items)} frames into {out_path}")
        return aligned_items
