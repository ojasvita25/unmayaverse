# Unmayaverse: Multimodal Video Frame Retrieval Engine

An end-to-end multimodal video retrieval system that searches video frames across **Visual features (SigLIP)**, **Dialogue transcripts (BGE)**, **Acoustic score stems (CLAP)**, and **Biometric character identification (InsightFace)**, indexed in a local **Qdrant Vector Database** and fused using **Weighted Reciprocal Rank Fusion (RRF)**.

---

## Benchmark Evaluation Results

Evaluated against the ground truth golden dataset (20 queries: 17 positive targets and 3 negative/out-of-domain rejection queries). Ground truth hits require retrieved timestamps to fall inside the exact target temporal interval.

### 1. Overall Benchmark Comparison Across Modes

| Retrieval Mode | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Negative Rejection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Visual Only** (SigLIP) | 0.3331 | 0.2353 | 0.3529 | 0.4706 | 0.6471 | 100.0% |
| **Visual + Dialogue Text** (SigLIP + BGE) | **0.6540** | **0.5882** | 0.6471 | 0.7059 | **0.9412** | **100.0%** |
| **Visual + Dialogue + Acoustic** (SigLIP + BGE + CLAP) | 0.6246 | 0.4706 | **0.7647** | **0.7647** | **0.9412** | **100.0%** |

* **100% Negative Rejection Accuracy**: All out-of-domain queries were completely rejected across all modes with zero false positives.
* **Top Rank Coverage**: Tri-modal search (`visual+text+audio`) achieves the highest **Recall@3** and **Recall@5** at **76.47%** (13 of 17 targets in Top 5).
* **Overall Recall**: Both bi-modal and tri-modal modes achieve **94.12% Recall@10** (16 of 17 positive targets successfully retrieved).

---

### 2. Modality Breakdown by Target Category

| Target Category | Queries | Visual MRR | Vis+Text MRR | Tri-Modal MRR | Visual R@5 | Vis+Text R@5 | Tri-Modal R@5 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Visual Targets** | 6 | **0.7417** | 0.7250 | 0.5278 | **1.0000** | 0.8333 | 0.6667 |
| **Semantic Targets** | 5 | 0.0000 | **1.0000** | 0.8667 | 0.0000 | **1.0000** | **1.0000** |
| **Multimodal Targets** | 6 | 0.2021 | 0.2946 | **0.5198** | 0.3333 | 0.3333 | **0.6667** |

* **Multimodal Gain**: On queries containing both visual actions and acoustic events, tri-modal fusion nearly doubles the MRR from `0.2946` to **`0.5198`**, and doubles Recall@5 from `33.3%` to **`66.7%`**.

---

### 3. Detailed Per-Query Retrieval Rankings Across Modes

| Query ID | Target Category | Query Description | Visual Only | Visual + Text | Visual + Text + Audio |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `q_01` | Visual | A woman drawing a question mark on a fogged bathroom mirror | Rank #1 | Rank #1 | Rank #2 |
| `q_02` | Semantic | A voiceover talking about seeking refuge in routines and a vice-like grip on normalcy | Missed | Rank #1 | Rank #1 |
| `q_03` | Multimodal | Friends in a cafe offering a woman a bite of honey cake while discussing festival preparation | Rank #2 | Rank #6 | Rank #7 |
| `q_04` | Semantic | A dancer talking about finding it really hard to get her body to do what she wants | Missed | Rank #1 | Rank #3 |
| `q_05` | Visual | Psychological horror scene of a woman looking at peeling skin on her left shoulder | Rank #4 | Rank #4 | Missed |
| `q_06` | Semantic | An echoing female voice telling the protagonist to be careful with what she wears | Missed | Rank #1 | Rank #1 |
| `q_07` | Visual | A dark, heavy snake coiled around a woman's torso in a dimly lit bathroom | Rank #1 | Rank #1 | Rank #1 |
| `q_08` | Semantic | An internal conversation about questioning every move so you never feel unsafe again | Missed | Rank #1 | Rank #1 |
| `q_09` | Multimodal | An announcer introducing a performer named Maya from India for the Figure8 dance festival | Missed | Rank #1 | Rank #1 |
| `q_10` | Visual | A woman performing fluid Tribal Fusion Belly Dance choreography on stage | Rank #1 | Rank #1 | Rank #1 |
| `q_11` | Semantic | A voiceover saying that she reminds herself to find joy in her own body and skin | Missed | Rank #1 | Rank #1 |
| `q_12` | Visual | Maya wiping a question mark off a bathroom mirror | Rank #5 | Rank #10 | Rank #6 |
| `q_13` | Visual | Maya hugging herself while looking at her reflection in front of a bathroom mirror | Rank #1 | Rank #1 | Rank #2 |
| `q_14` | Multimodal | A woman looking into a fogged bathroom mirror while a water tap runs continuously in the background | Rank #9 | Missed | **Rank #2** |
| `q_15` | Multimodal | A belly dancer performing on stage to a soundtrack featuring high-pitched vocal ululations | Rank #8 | Rank #8 | **Rank #7** |
| `q_16` | Multimodal | A dancer performing fluid choreography on stage driven by heavy tribal beats and rising percussion | Rank #3 | Rank #3 | **Rank #1** |
| `q_17` | Multimodal | Gentle ambient music accompanying a woman wiping a question mark off the bathroom mirror and embracing her reflection | Rank #7 | Rank #7 | **Rank #3** |
| `q_neg_01` | Negative | A high-speed car chase through neon-lit futuristic city streets at night | Rejected | Rejected | Rejected |
| `q_neg_02` | Negative | A chef chopping fresh vegetables and stirring a boiling pot of soup in a commercial kitchen | Rejected | Rejected | Rejected |
| `q_neg_03` | Negative | An astronaut floating in zero gravity outside the International Space Station | Rejected | Rejected | Rejected |

#### Acoustic Grounding Highlights
* **`q_14` (Running water tap)**: Missed completely in bi-modal mode, but lifted to **Rank #2** in tri-modal mode via CLAP background audio grounding.
* **`q_16` (Tribal beats & rising percussion)**: Promoted from Rank #3 to **Rank #1**.
* **`q_17` (Gentle ambient music)**: Promoted from Rank #7 to **Rank #3**.
* **`q_15` (High-pitched vocal ululations)**: Gained entry into the top ranks at **Rank #7**.

---

## Pipeline Architecture

1. **Preprocessing & Feature Extraction**:
   * **Visual (SigLIP SO400M / 1152-d)**: 1 FPS keyframes downscaled to 720p with JPEG quality 3 (~35KB/frame).
   * **Semantic (BGE Large v1.5 / 1024-d)**: Transcripts generated by Whisper with timestamp alignment.
   * **Acoustic (CLAP HTSAT / 512-d)**: Demucs vocal separation (`htdemucs`) isolates background score and percussion stems.
   * **Face Verification (InsightFace buffalo_l / 512-d)**: Biometric cosine matching ($\ge 0.35$) tags frames containing Maya.
2. **Indexing & Retrieval**:
   * Local embedded **Qdrant Vector Database** storing named multi-vectors (`visual`, `semantic`, `audio`).
   * **Weighted Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF}(i) = \sum_{m \in M} w_m \cdot \frac{1}{K + \text{rank}_m(i)}, \quad K = 20$$
   * Individual modality noise floors (`Visual: 0.10`, `Semantic: 0.45`, `Acoustic: 0.20`) and minimum RRF threshold (`MIN_RRF_SCORE = 0.015`).

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/ojasvita25/unmayaverse.git
cd unmayaverse
pip install -r requirements.txt

# Extract frames, audio separation, transcripts, and face tags
python -m src.main --preprocess

# Generate embeddings and populate Qdrant DB
python -m src.main --embed

# Run retrieval benchmark evaluation
python -m src.main --evaluate --mode all

# Launch interactive Gradio Web UI
python -m src.main --ui
