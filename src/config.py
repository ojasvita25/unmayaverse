"""
Unmayaverse: Centralized Configuration & Hyperparameters
Multimodal AI Video Frame Retrieval Framework
"""

import os

# =============================================================================
# DIRECTORY & FILE PATH CONFIGURATIONS
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_FOLDER_PATH = os.path.join(PROJECT_ROOT, "data")
QDRANT_DB_DIR = os.path.join(MEDIA_FOLDER_PATH, "qdrant_db")

# Media & Subtitle Files
VIDEO_FILE = "unmaya_HD.mp4"
SRT_FILE = "unmaya_subtitles.srt"
MASTER_AUDIO_FILE = "master_audio.wav"
FRAMES_DIR_NAME = "frames_1fps"

# Processed Metadata & Embeddings Artifacts
TRANSCRIPT_JSON = "unmaya_semantic_transcripts.json"
TEXT_METADATA_JSON = "text_metadata.json"
FRAME_METADATA_JSON = "unmaya_frame_metadata.json"
FACE_EMBEDDINGS_JSON = "unmaya_face_embeddings.json"
CHARACTER_CLUSTERS_JSON = "character_clusters.json"

IMAGE_EMBEDDINGS_NPY = "unmaya_image_embeddings.npy"
TEXT_EMBEDDINGS_NPY = "unmaya_text_embeddings.npy"
AUDIO_EMBEDDINGS_NPY = "unmaya_audio_embeddings.npy"

# Golden Dataset for Evaluation
GOLDEN_DATASET_FILE = "unmayaverse_golden_dataset.json"

# =============================================================================
# MODEL IDENTIFIERS
# =============================================================================
VISUAL_MODEL_NAME = "google/siglip-so400m-patch14-384"
SEMANTIC_MODEL_NAME = "BAAI/bge-large-en-v1.5"
ACOUSTIC_MODEL_NAME = "laion/clap-htsat-unfused"
FACE_MODEL_NAME = "buffalo_l"

# =============================================================================
# VECTOR EMBEDDING DIMENSIONS
# =============================================================================
SIGLIP_VECTOR_DIM = 1152
BGE_VECTOR_DIM = 1024
CLAP_VECTOR_DIM = 512
INSIGHTFACE_VECTOR_DIM = 512

# =============================================================================
# QDRANT VECTOR DB CONFIGURATION
# =============================================================================
QDRANT_COLLECTION_NAME = "unmaya_multimodal_frames"

# =============================================================================
# PREPROCESSING & INGESTION DEFAULTS
# =============================================================================
DEFAULT_FPS = 1
DEFAULT_WHISPER_MODEL = "medium"
DEFAULT_REFERENCE_FRAME = 26
DEFAULT_FACE_THRESHOLD = 0.35
FRAME_MAX_WIDTH = 720          # Max width for extracted frames (preserves aspect ratio)
FRAME_JPEG_QUALITY = 3         # FFmpeg q:v quality (2=near-lossless, 3-4=optimal ~50KB/frame)

# =============================================================================
# RETRIEVAL, FUSION & FILTERING HYPERPARAMETERS
# =============================================================================
RRF_K_CONSTANT = 60
VISUAL_NOISE_FLOOR = 0.15
SEMANTIC_NOISE_FLOOR = 0.20
ACOUSTIC_NOISE_FLOOR = 0.10
DEFAULT_MIN_Z_SCORE = 1.8

DEFAULT_TOP_K = 5
DEFAULT_VIS_WEIGHT = 0.33
DEFAULT_SEM_WEIGHT = 0.33
DEFAULT_AUD_WEIGHT = 0.34

# =============================================================================
# UI & REMOTE STORAGE DEFAULTS
# =============================================================================
DEFAULT_GRADIO_PORT = 7860

# =============================================================================
# SEARCH MODES
# =============================================================================
SEARCH_MODE_VISUAL = "visual"
SEARCH_MODE_VISUAL_TEXT = "visual+text"
SEARCH_MODE_VISUAL_TEXT_AUDIO = "visual+text+audio"

SEARCH_MODES = [
    SEARCH_MODE_VISUAL,
    SEARCH_MODE_VISUAL_TEXT,
    SEARCH_MODE_VISUAL_TEXT_AUDIO,
]
