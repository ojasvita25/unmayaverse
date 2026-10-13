"""
Unmayaverse: Interactive Gradio Search & Retrieval Application
Provides web interface for multimodal video frame search with automatic identity detection.
"""

import os
import time
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
    DEFAULT_WEIGHTS_VIS_TEXT,
    DEFAULT_WEIGHTS_TRI_MODAL,
    DEFAULT_GRADIO_PORT,
    MIN_RRF_SCORE,
)
from .search_engine import MultimodalSearchEngine


def build_gradio_ui(search_engine: MultimodalSearchEngine) -> gr.Blocks:
    """Builds the Gradio Web Application for Frame Search & Retrieval."""

    def search_wrapper(
        query: str,
        mode: str,
        top_k: int,
        min_rrf: float,
        vis_w: float,
        sem_w: float,
        aud_w: float
    ) -> Tuple[List[Any], str]:
        if not query.strip():
            return [], "Please enter a search query."

        # Measure search & retrieval execution latency
        start_time = time.time()
        results = search_engine.search(
            query_text=query,
            mode=mode,
            top_k=int(top_k),
            min_rrf_score=float(min_rrf),
            vis_weight=float(vis_w),
            sem_weight=float(sem_w),
            aud_weight=float(aud_w)
        )
        latency_ms = (time.time() - start_time) * 1000.0

        if not results:
            return [], f"No frames matched under mode '{mode}' ({latency_ms:.1f}ms)."

        gallery_items = []
        for r in results:
            raw_path = r.get("file_path", "")
            frame_idx = r.get("frame_index", 0)
            
            # Caption now includes score, video timestamp, and search latency
            caption = f"Frame {frame_idx} | Score: {r['score']:.4f} | Time: {r['timestamp']}s | Latency: {latency_ms:.1f}ms"
            if r.get("transcript"):
                caption += f"\n\"{r['transcript'][:60]}...\""
            if r.get("characters"):
                caption += f"\n[{', '.join(r['characters'])}]"

            # Dynamically resolve path across environments (Local, Colab, or Docker)
            resolved_path = raw_path
            if not os.path.exists(resolved_path):
                filename = os.path.basename(raw_path) if raw_path else f"frame_{frame_idx+1:04d}.jpg"
                candidate = os.path.join(MEDIA_FOLDER_PATH, "frames_1fps", filename)
                if os.path.exists(candidate):
                    resolved_path = candidate

            # Load actual image pixels into memory for Gradio Gallery
            if os.path.exists(resolved_path):
                img = Image.open(resolved_path).convert("RGB")
                gallery_items.append((img, caption))
            else:
                placeholder = Image.new("RGB", (384, 216), color=(45, 52, 58))
                gallery_items.append((placeholder, caption))

        summary = f"Retrieved {len(results)} frame(s) in {latency_ms:.1f}ms using mode: {mode}."
        if "maya" in query.lower():
            summary += " (Auto-detected character: 'Maya' via InsightFace filter)."

        return gallery_items, summary


    with gr.Blocks(title="Unmayaverse Multimodal AI Video Search") as demo:
        gr.Markdown(
            """
            # Unmayaverse: Multimodal AI Video Search
            Search video frames across Visual (SigLIP), Dialogue Text (BGE), Acoustic Score (CLAP), and Automatic Character Identity (InsightFace).
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g. Maya drawing a question mark on a bathroom mirror",
                    lines=2
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
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=20, step=1, 
                        value=DEFAULT_TOP_K, label="Top K Results"
                    )

                with gr.Accordion("Modality RRF Weights Tuning", open=False):
                    min_rrf_slider = gr.Slider(
                        minimum=0.0, maximum=0.040, step=0.001, 
                        value=MIN_RRF_SCORE, label="Min Fused RRF Threshold"
                    )
                    vis_w_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, 
                        value=DEFAULT_WEIGHTS_TRI_MODAL["visual"], 
                        label="Visual Weight (SigLIP)"
                    )
                    sem_w_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, 
                        value=DEFAULT_WEIGHTS_TRI_MODAL["semantic"], 
                        label="Semantic Weight (BGE Dialogue)"
                    )
                    aud_w_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, 
                        value=DEFAULT_WEIGHTS_TRI_MODAL["audio"], 
                        label="Acoustic Weight (CLAP Audio Score)"
                    )


                # Helper to distribute remaining weight proportionally between 2 sliders
                def balance_two(changed_val, current_other1, current_other2):
                    rem = max(0.0, round(1.0 - float(changed_val), 2))
                    other_sum = float(current_other1) + float(current_other2)
                    if other_sum > 0:
                        val1 = round(rem * (float(current_other1) / other_sum), 2)
                        val2 = round(rem - val1, 2)
                    else:
                        val1 = round(rem / 2.0, 2)
                        val2 = round(rem - val1, 2)
                    return max(0.0, val1), max(0.0, val2)

                # Real-time auto-balancing when user drags a slider
                def on_vis_input(vis_val, sem_val, aud_val, mode):
                    if mode == SEARCH_MODE_VISUAL_TEXT:
                        sem_new = max(0.0, round(1.0 - float(vis_val), 2))
                        return gr.update(value=sem_new), gr.update(value=0.0)
                    elif mode == SEARCH_MODE_VISUAL_TEXT_AUDIO:
                        s_new, a_new = balance_two(vis_val, sem_val, aud_val)
                        return gr.update(value=s_new), gr.update(value=a_new)
                    return gr.update(), gr.update()

                def on_sem_input(sem_val, vis_val, aud_val, mode):
                    if mode == SEARCH_MODE_VISUAL_TEXT:
                        vis_new = max(0.0, round(1.0 - float(sem_val), 2))
                        return gr.update(value=vis_new), gr.update(value=0.0)
                    elif mode == SEARCH_MODE_VISUAL_TEXT_AUDIO:
                        v_new, a_new = balance_two(sem_val, vis_val, aud_val)
                        return gr.update(value=v_new), gr.update(value=a_new)
                    return gr.update(), gr.update()

                def on_aud_input(aud_val, vis_val, sem_val, mode):
                    if mode == SEARCH_MODE_VISUAL_TEXT_AUDIO:
                        v_new, s_new = balance_two(aud_val, vis_val, sem_val)
                        return gr.update(value=v_new), gr.update(value=s_new)
                    return gr.update(), gr.update()

                vis_w_slider.input(
                    fn=on_vis_input,
                    inputs=[vis_w_slider, sem_w_slider, aud_w_slider, mode_selector],
                    outputs=[sem_w_slider, aud_w_slider]
                )
                sem_w_slider.input(
                    fn=on_sem_input,
                    inputs=[sem_w_slider, vis_w_slider, aud_w_slider, mode_selector],
                    outputs=[vis_w_slider, aud_w_slider]
                )
                aud_w_slider.input(
                    fn=on_aud_input,
                    inputs=[aud_w_slider, vis_w_slider, sem_w_slider, mode_selector],
                    outputs=[vis_w_slider, sem_w_slider]
                )

                # Dynamically update weight sliders when search mode changes
                def on_mode_change(selected_mode):
                    if selected_mode == SEARCH_MODE_VISUAL_TEXT:
                        return (
                            gr.update(value=DEFAULT_WEIGHTS_VIS_TEXT["visual"], interactive=True),
                            gr.update(value=DEFAULT_WEIGHTS_VIS_TEXT["semantic"], interactive=True),
                            gr.update(value=0.0, interactive=False),
                        )
                    elif selected_mode == SEARCH_MODE_VISUAL_TEXT_AUDIO:
                        return (
                            gr.update(value=DEFAULT_WEIGHTS_TRI_MODAL["visual"], interactive=True),
                            gr.update(value=DEFAULT_WEIGHTS_TRI_MODAL["semantic"], interactive=True),
                            gr.update(value=DEFAULT_WEIGHTS_TRI_MODAL["audio"], interactive=True),
                        )
                    else:  # Visual Only
                        return (
                            gr.update(value=1.0, interactive=False),
                            gr.update(value=0.0, interactive=False),
                            gr.update(value=0.0, interactive=False),
                        )

                mode_selector.change(
                    fn=on_mode_change,
                    inputs=[mode_selector],
                    outputs=[vis_w_slider, sem_w_slider, aud_w_slider]
                )

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
            top_k_slider,
            min_rrf_slider,
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
    demo.launch(server_port=args.port, share=args.share, allowed_paths=[args.media_folder])


if __name__ == "__main__":
    main()
