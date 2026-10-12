"""
Unmayaverse: Interactive Gradio Search & Retrieval Application
Provides web interface for multimodal video frame search.
"""

import os
import argparse
from typing import List, Tuple, Any
from PIL import Image
import gradio as gr

from .config import (
    MEDIA_FOLDER_PATH,
    SEARCH_MODE_VISUAL,
    SEARCH_MODE_VISUAL_TEXT,
    SEARCH_MODE_VISUAL_TEXT_AUDIO,
    DEFAULT_TOP_K,
    DEFAULT_VIS_WEIGHT,
    DEFAULT_SEM_WEIGHT,
    DEFAULT_AUD_WEIGHT,
    DEFAULT_GRADIO_PORT,
)
from .search_engine import MultimodalSearchEngine


def build_gradio_ui(search_engine: MultimodalSearchEngine) -> gr.Blocks:
    """Builds the Gradio Web Application for Frame Search & Retrieval."""

    def get_character_options():
        return search_engine.get_available_characters()

    def search_wrapper(
        query: str,
        mode: str,
        character: str,
        top_k: int,
        min_z: float,
        vis_w: float,
        sem_w: float,
        aud_w: float
    ) -> Tuple[List[Any], str]:
        if not query.strip():
            return [], "Please enter a search query."

        results = search_engine.search(
            query_text=query,
            mode=mode,
            character_name=character,
            top_k=int(top_k),
            min_z_score=float(min_z),
            vis_weight=float(vis_w),
            sem_weight=float(sem_w),
            aud_weight=float(aud_w)
        )

        if not results:
            return [], f"No frames passed confidence threshold under mode '{mode}' with min Z-score >= {min_z}."

        gallery_items = []
        for r in results:
            img_path = r.get("file_path", "")
            caption = f"Frame {r['frame_index']} | Score: {r['score']:.4f} | Time: {r['timestamp']}s"
            if r.get("transcript"):
                caption += f"\n\"{r['transcript'][:60]}...\""
            if r.get("characters"):
                caption += f"\n[{', '.join(r['characters'])}]"

            if os.path.exists(img_path):
                gallery_items.append((img_path, caption))
            else:
                placeholder = Image.new("RGB", (384, 216), color=(45, 52, 58))
                gallery_items.append((placeholder, caption))

        summary = f"Retrieved {len(results)} frame(s) using mode: {mode}."
        if character not in ["All Characters", "All Characters (Auto-Detect)"]:
            summary += f" Filtered by character: '{character}'."

        return gallery_items, summary

    with gr.Blocks(title="Unmayaverse Multimodal AI Video Search") as demo:
        gr.Markdown(
            """
            # Unmayaverse: Multimodal AI Video Search
            Search video frames across Visual (SigLIP), Dialogue Text (BGE), Acoustic Score (CLAP), and Character Identity (InsightFace).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. Maya drawing a question mark on a bathroom mirror",
                    lines=2
                )
                character_dropdown = gr.Dropdown(
                    choices=get_character_options(),
                    value="All Characters (Auto-Detect)",
                    label="Character Filter"
                )
                mode_selector = gr.Radio(
                    choices=[
                        ("Visual Only", SEARCH_MODE_VISUAL),
                        ("Visual + Dialogue Text", SEARCH_MODE_VISUAL_TEXT),
                        ("Visual + Dialogue + Acoustic Score", SEARCH_MODE_VISUAL_TEXT_AUDIO)
                    ],
                    value=SEARCH_MODE_VISUAL_TEXT_AUDIO,
                    label="Search Mode"
                )

                with gr.Row():
                    top_k_slider = gr.Slider(minimum=1, maximum=20, step=1, value=DEFAULT_TOP_K, label="Top K Results")
                    min_z_slider = gr.Slider(minimum=0.0, maximum=3.0, step=0.1, value=0.0, label="Min Z-Score Threshold")

                with gr.Accordion("Modality RRF Weights Tuning", open=False):
                    vis_w_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_VIS_WEIGHT, label="Visual Weight (SigLIP)")
                    sem_w_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_SEM_WEIGHT, label="Semantic Weight (BGE Dialogue)")
                    aud_w_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_AUD_WEIGHT, label="Acoustic Weight (CLAP Audio Score)")

                search_btn = gr.Button("Search Video Frames", variant="primary")

            with gr.Column(scale=2):
                status_output = gr.Textbox(label="Status / Retrieval Summary", interactive=False)
                gallery = gr.Gallery(
                    label="Retrieved Video Frames",
                    columns=[3],
                    rows=[2],
                    object_fit="contain",
                    height="auto"
                )

        search_inputs = [
            query_input,
            mode_selector,
            character_dropdown,
            top_k_slider,
            min_z_slider,
            vis_w_slider,
            sem_w_slider,
            aud_w_slider
        ]
        search_btn.click(fn=search_wrapper, inputs=search_inputs, outputs=[gallery, status_output])
        query_input.submit(fn=search_wrapper, inputs=search_inputs, outputs=[gallery, status_output])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Unmayaverse Gradio Web UI")
    parser.add_argument("--media-folder", type=str, default=MEDIA_FOLDER_PATH, help="Path to media directory")
    parser.add_argument("--port", type=int, default=DEFAULT_GRADIO_PORT, help="Port to run Gradio server on")
    parser.add_argument("--share", action="store_true", help="Generate public share link")

    args = parser.parse_args()

    engine = MultimodalSearchEngine(media_folder_path=args.media_folder)
    demo = build_gradio_ui(engine)
    print(f"Launching Gradio UI on port {args.port}...")
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
