"""Cinderworks Studio — Gradio Blocks shell.

Thin shell that wires Gradio UI components to handlers. Contains NO
inference logic, no download logic, no SQL, no direct backend imports.
All model access goes through the registry (via handlers).

Implements: Requirements 1.4, 2.1, 2.4, 4.2, 8.3, 8.4, 8.5, 8.6, 9.1, 10.1
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _load_history_to_dataframe(page: int):
    """Convert handler output to dataframe rows and increment page.

    Displays: truncated prompt (120 chars), seed, steps, resolution,
    precision, and model_id. Returns (rows, next_page).
    """
    jobs = handlers.on_load_history(page=page)
    if jobs and "error" in jobs[0]:
        return [[jobs[0]["error"], "", "", "", "", "", ""]], page
    rows = []
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
    if not rows:
        rows = [["", "No jobs found", "", "", "", "", ""]]
    return rows, page + 1


def _load_first_page():
    """Load the first page of history (called on tab select)."""
    rows, _ = _load_history_to_dataframe(0)
    return rows, 0


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
                    load_more_btn = gr.Button("Load More", variant="secondary")
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
                    delete_job_btn = gr.Button(
                        "Delete Job", variant="stop"
                    )
                    load_params_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        placeholder="",
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

        # History: auto-load first page when tab is selected
        history_tab.select(
            fn=_load_first_page,
            inputs=[],
            outputs=[history_display, history_page_state],
        )

        # History: load more (next page)
        load_more_btn.click(
            fn=_load_history_to_dataframe,
            inputs=[history_page_state],
            outputs=[history_display, history_page_state],
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

        # Delete job: remove from DB and refresh history
        def _on_delete_job_click(job_id):
            result = handlers.on_delete_job(int(job_id) if job_id else 0)
            # Refresh history after deletion
            rows, _ = _load_history_to_dataframe(0)
            return result, rows, 0

        delete_job_btn.click(
            fn=_on_delete_job_click,
            inputs=[selected_job_id],
            outputs=[load_params_status, history_display, history_page_state],
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
