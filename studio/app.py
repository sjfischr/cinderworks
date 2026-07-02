"""Cinderworks Studio — Gradio Blocks shell.

Thin shell that wires Gradio UI components to handlers. Contains NO
inference logic, no download logic, no SQL, no direct backend imports.
All model access goes through the registry (via handlers).

Implements: Requirements 1.4, 2.1, 2.4, 4.2, 8.3, 8.4, 8.5, 8.6, 9.1, 10.1
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Configure the CUDA caching allocator BEFORE anything imports torch.
# expandable_segments avoids the repeated cudaMalloc/cudaFree churn that
# per-block transient allocations (layerwise fp8 upcasts, attention
# buffers) otherwise cause — on Windows/WDDM each raw cudaMalloc can
# stall on driver paging. Respects an existing user-set value; torch
# ignores the option with a warning on platforms that don't support it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Ensure the repo root (parent of studio/) is on sys.path so that
# `import studio.*` works regardless of the working directory.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import gradio as gr

from studio.ui.theme import CUSTOM_CSS, get_theme
from studio.ui import controls, handlers
from studio.core import system_check


# ---------------------------------------------------------------------------
# History tab helpers
# ---------------------------------------------------------------------------

_HISTORY_HEADERS = [
    "ID", "Prompt", "Seed", "Steps", "Resolution", "Precision", "Model"
]
_HISTORY_DATATYPES = ["number", "str", "number", "number", "str", "str", "str"]

# Placeholder text for missing images
_IMAGE_PLACEHOLDER = "[image unavailable]"


def _render_history_page(page: int):
    """Render one page of history.

    Returns (rows, page, page_label_markdown, checkbox_update) where the
    checkbox update carries the current page's job IDs as selectable
    choices for multi-delete.
    """
    jobs = handlers.on_load_history(page=page)
    if jobs and "error" in jobs[0]:
        return (
            [[jobs[0]["error"], "", "", "", "", "", ""]],
            page,
            f"Page {page + 1}",
            gr.update(choices=[], value=[]),
        )
    rows = []
    choices = []
    for j in jobs:
        params = j.get("params", {})
        w = params.get("width", "?")
        h = params.get("height", "?")
        resolution = f"{w}\u00d7{h}"
        rows.append([
            j.get("id", ""),
            j.get("prompt", "")[:120],
            j.get("seed", ""),
            params.get("steps", ""),
            resolution,
            params.get("precision", ""),
            j.get("model_id", ""),
        ])
        choices.append(
            (f"#{j.get('id')} \u2014 {j.get('prompt', '')[:48]}", str(j.get("id")))
        )
    if not rows:
        rows = [["", "No jobs found", "", "", "", "", ""]]
    return rows, page, f"Page {page + 1}", gr.update(choices=choices, value=[])


def _load_first_page():
    """Load the first page of history (called on tab select)."""
    return _render_history_page(0)


def _history_prev_page(page: int):
    """Go to the previous history page (clamped at the first)."""
    return _render_history_page(max(0, (page or 0) - 1))


def _history_next_page(page: int):
    """Go to the next history page; stay put if it would be empty."""
    next_page = (page or 0) + 1
    jobs = handlers.on_load_history(page=next_page)
    if not jobs or (jobs and "error" in jobs[0]):
        return _render_history_page(page or 0)
    return _render_history_page(next_page)


def _on_load_params_click(selected_job_id):
    """Load parameters from a job and return values for Generate tab fields.

    Returns a tuple:
        (prompt, seed, steps, width, height, precision, batch_size, batch_count, status_msg)

    If the job_id is invalid or loading fails, returns gr.update() placeholders
    with an error status message.
    """
    if not selected_job_id or selected_job_id <= 0:
        return (
            gr.update(),  # prompt
            gr.update(),  # seed
            gr.update(),  # steps
            gr.update(),  # width
            gr.update(),  # height
            gr.update(),  # precision
            gr.update(),  # batch_size
            gr.update(),  # batch_count
            "No job selected.",
        )

    result = handlers.on_load_params(int(selected_job_id))
    if "error" in result:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            result["error"],
        )

    return (
        result["prompt"],
        result["seed"],
        result["steps"],
        result["width"],
        result["height"],
        result["precision"],
        result["batch_size"],
        result["batch_count"],
        f"\u2705 Parameters loaded for Job #{int(selected_job_id)}",
    )


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Construct the full Cinderworks Studio UI.

    Stays thin: wiring only. No inference, no download, no SQL here.
    On load, fires a CUDA/readiness check and sets the banner state.
    """
    with gr.Blocks(
        theme=get_theme(),
        css=CUSTOM_CSS,
        title="Cinderworks Studio",
    ) as app:

        # --- Readiness Banner (top-level, above tabs) ---
        readiness_banner = gr.Markdown(
            value="",
            visible=False,
            elem_id="readiness-banner",
        )

        # --- Tabbed Layout ---
        with gr.Tabs():

            # =============================================================
            # GENERATE TAB
            # =============================================================
            with gr.Tab("Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_box = controls.create_prompt_controls()
                        steps, seed, width, height = controls.create_sampler_controls()
                        precision = controls.create_precision_picker()
                        precision_warning = gr.Markdown(
                            value="",
                            elem_id="precision-warning",
                        )
                        batch_size, batch_count = controls.create_batch_controls()
                        generate_btn = gr.Button(
                            "Generate", variant="primary", size="lg"
                        )
                    with gr.Column(scale=3):
                        progress_output = gr.Textbox(
                            label="Progress",
                            interactive=False,
                            lines=3,
                            placeholder="Generation progress will appear here...",
                        )
                        image_gallery = gr.Gallery(
                            label="Results",
                            columns=2,
                            height="auto",
                        )

            # =============================================================
            # HISTORY TAB
            # =============================================================
            with gr.Tab("History") as history_tab:
                with gr.Column():
                    history_display = gr.Dataframe(
                        headers=_HISTORY_HEADERS,
                        datatype=_HISTORY_DATATYPES,
                        label="Generation History",
                        interactive=False,
                    )
                    with gr.Row():
                        history_prev_btn = gr.Button("◀ Previous", variant="secondary")
                        history_page_label = gr.Markdown("Page 1")
                        history_next_btn = gr.Button("Next ▶", variant="secondary")
                    gr.Markdown("---")
                    gr.Markdown("#### Delete Jobs")
                    history_delete_select = gr.CheckboxGroup(
                        label="Select jobs to delete (removes database "
                        "entries AND image files)",
                        choices=[],
                    )
                    delete_selected_btn = gr.Button(
                        "Delete Selected", variant="stop"
                    )
                    gr.Markdown("---")
                    gr.Markdown("#### Load Parameters from Job")
                    selected_job_id = gr.Number(
                        label="Job ID to load",
                        precision=0,
                        minimum=1,
                        info="Enter the Job ID from the table above, then click Load Parameters.",
                    )
                    load_params_btn = gr.Button(
                        "Load Parameters to Generate Tab", variant="secondary"
                    )
                    load_params_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        placeholder="",
                    )

            # =============================================================
            # UPSCALE TAB
            # =============================================================
            with gr.Tab("Upscale"):
                from studio.models import upscale as _upscale_mod

                with gr.Row():
                    with gr.Column():
                        upscale_input = gr.Image(
                            label="Image to upscale",
                            type="filepath",
                        )
                        upscale_method = gr.Radio(
                            choices=_upscale_mod.list_methods(),
                            value=_upscale_mod.METHOD_LANCZOS,
                            label="Method",
                            info="Lanczos needs no model. Real-ESRGAN is "
                            "sharper but needs a one-time ~67 MB download "
                            "(Models tab).",
                        )
                        upscale_scale = gr.Slider(
                            minimum=1.0,
                            maximum=4.0,
                            value=2.0,
                            step=0.5,
                            label="Scale factor",
                        )
                        upscale_btn = gr.Button("Upscale", variant="primary")
                        upscale_status = gr.Textbox(
                            label="Status", interactive=False, lines=1
                        )
                    with gr.Column():
                        upscale_output = gr.Image(
                            label="Upscaled result",
                            type="filepath",
                            interactive=False,
                        )

            # =============================================================
            # MODELS TAB
            # =============================================================
            with gr.Tab("Models"):
                with gr.Column():
                    gr.Markdown("### Model Management")
                    model_status_display = gr.Markdown(
                        value="Checking model status...",
                        elem_id="model-status",
                    )
                    check_status_btn = gr.Button(
                        "Check Status", variant="secondary"
                    )
                    download_btn = gr.Button(
                        "Download Krea 2 Turbo", variant="primary"
                    )
                    download_upscaler_btn = gr.Button(
                        "Download Upscaler (Real-ESRGAN 4x, ~67 MB)",
                        variant="secondary",
                    )
                    download_progress = gr.Textbox(
                        label="Download Progress",
                        interactive=False,
                        lines=6,
                        placeholder="Download progress will appear here...",
                    )

            # =============================================================
            # SETTINGS TAB
            # =============================================================
            with gr.Tab("Settings"):
                gr.Markdown(
                    "### Settings\n\n"
                    "Settings will be available in a future update."
                )

        # -----------------------------------------------------------------
        # STATE (hidden) for pagination
        # -----------------------------------------------------------------
        history_page_state = gr.State(value=0)

        # -----------------------------------------------------------------
        # EVENT WIRING
        # -----------------------------------------------------------------

        # Generate button -> on_generate handler
        generate_btn.click(
            fn=handlers.on_generate,
            inputs=[
                prompt_box,
                steps,
                seed,
                width,
                height,
                precision,
                batch_size,
                batch_count,
            ],
            outputs=[progress_output, image_gallery],
        )

        # Precision picker -> advisory VRAM warning (hard refusal still
        # happens at generate time; this warns before the click)
        precision.change(
            fn=handlers.on_precision_change,
            inputs=[precision],
            outputs=[precision_warning],
        )

        # Check Status button -> refresh model status display
        def _get_model_status_text():
            """Return markdown showing per-file model download state."""
            from studio.models import downloader
            return downloader.get_model_info_text("krea2-turbo")

        check_status_btn.click(
            fn=_get_model_status_text,
            inputs=[],
            outputs=[model_status_display],
        )

        # Download button -> on_download handler (generator streams progress)
        # After download completes, re-evaluate readiness and update banner + model status
        def _refresh_after_download():
            """Re-evaluate readiness and return updated banner + model status."""
            banner = system_check.get_readiness_banner()
            from studio.models import downloader
            status_text = downloader.get_model_info_text("krea2-turbo")
            return (
                gr.update(
                    visible=banner.get("visible", False),
                    value=banner.get("value", ""),
                ),
                status_text,
            )

        download_btn.click(
            fn=handlers.on_download,
            inputs=[],
            outputs=[download_progress],
        ).then(
            fn=_refresh_after_download,
            inputs=[],
            outputs=[readiness_banner, model_status_display],
        )

        # Upscaler model download
        download_upscaler_btn.click(
            fn=handlers.on_download_upscaler,
            inputs=[],
            outputs=[download_progress],
        )

        # Upscale an image
        upscale_btn.click(
            fn=handlers.on_upscale,
            inputs=[upscale_input, upscale_method, upscale_scale],
            outputs=[upscale_output, upscale_status],
        )

        # History: auto-load first page when tab is selected
        _history_outputs = [
            history_display,
            history_page_state,
            history_page_label,
            history_delete_select,
        ]
        history_tab.select(
            fn=_load_first_page,
            inputs=[],
            outputs=_history_outputs,
        )

        # History: page navigation
        history_prev_btn.click(
            fn=_history_prev_page,
            inputs=[history_page_state],
            outputs=_history_outputs,
        )
        history_next_btn.click(
            fn=_history_next_page,
            inputs=[history_page_state],
            outputs=_history_outputs,
        )

        # Load params: populate Generate tab fields from a past job
        load_params_btn.click(
            fn=_on_load_params_click,
            inputs=[selected_job_id],
            outputs=[
                prompt_box,
                seed,
                steps,
                width,
                height,
                precision,
                batch_size,
                batch_count,
                load_params_status,
            ],
        )

        # Delete selected jobs: remove DB rows + image files, re-render page
        def _on_delete_selected_click(selected_ids, page):
            result = handlers.on_delete_jobs(
                [int(s) for s in (selected_ids or [])]
            )
            rows, page, label, checkbox = _render_history_page(page or 0)
            return result, rows, page, label, checkbox

        delete_selected_btn.click(
            fn=_on_delete_selected_click,
            inputs=[history_delete_select, history_page_state],
            outputs=[load_params_status, *_history_outputs],
        )

        # -----------------------------------------------------------------
        # ON LOAD: Set initial readiness banner state and model status
        # -----------------------------------------------------------------

        def _startup_banner():
            """Run system check on load and return banner state + model status."""
            system_check.startup_cuda_check()
            banner = system_check.get_readiness_banner()
            from studio.models import downloader
            status_text = downloader.get_model_info_text("krea2-turbo")
            return (
                gr.update(
                    visible=banner.get("visible", False),
                    value=banner.get("value", ""),
                ),
                status_text,
            )

        app.load(fn=_startup_banner, outputs=[readiness_banner, model_status_display])

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

app = build_app()

if __name__ == "__main__":
    app.launch()
