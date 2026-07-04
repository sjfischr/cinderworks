"""UI Gallery Component — Gallery with send-to action buttons and keyboard navigation.

Provides a Gradio Gallery component with action buttons for sending selected
images to img2img or upscale workflows. Supports both single-image and
multi-image batch results. Includes client-side JavaScript for ArrowLeft/
ArrowRight keyboard navigation with clamped boundary behavior.

The handler wiring (what happens when buttons are clicked) is done in a
later integration task — this module creates and returns the components.

Implements: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 9.1, 9.2, 9.3
"""

from __future__ import annotations

import gradio as gr


# Client-side JavaScript for gallery keyboard navigation.
# Handles ArrowLeft/ArrowRight to move between thumbnails with clamped
# boundary behavior (no wrap-around). Adds a 2px solid accent border as
# the visual focus indicator on the focused thumbnail. Initializes focus
# to the first image when the gallery receives focus with no prior selection.
# Keyboard shortcuts: 'i' sends focused image to img2img, 'u' sends to upscale.
GALLERY_KEYBOARD_JS = """
<script>
(function() {
    "use strict";

    const GALLERY_ID = "results-gallery";
    const FOCUS_STYLE_CLASS = "kiro-gallery-focused";

    // Inject focus indicator CSS (2px solid accent border)
    const style = document.createElement("style");
    style.textContent = `
        .${FOCUS_STYLE_CLASS} {
            outline: none !important;
            border: 2px solid var(--color-accent, #f59e0b) !important;
            border-radius: 4px;
            box-sizing: border-box;
        }
    `;
    document.head.appendChild(style);

    let focusedIndex = -1;

    function getGalleryContainer() {
        return document.getElementById(GALLERY_ID);
    }

    function getThumbnails(container) {
        if (!container) return [];
        // Gradio gallery thumbnails are typically buttons or clickable elements
        // inside the gallery grid. They may be <button>, <img>, or wrapper divs.
        const thumbs = container.querySelectorAll(
            ".thumbnail-item, .gallery-item, [data-testid='thumbnail'], .thumbnails button, .grid-wrap .thumbnail-lg, .preview .thumbnails button, .thumbnails .thumbnail-small, .thumbnail-small"
        );
        if (thumbs.length > 0) return Array.from(thumbs);

        // Fallback: try common Gradio gallery structures
        const gridItems = container.querySelectorAll(".grid-container button, .grid-wrap button, .gallery-item img");
        if (gridItems.length > 0) return Array.from(gridItems);

        // Last fallback: any clickable image containers in the gallery
        const imgContainers = container.querySelectorAll("button:has(img), .thumbnail-item, [role='button']");
        return Array.from(imgContainers);
    }

    function clearFocusIndicators(thumbnails) {
        thumbnails.forEach(function(thumb) {
            thumb.classList.remove(FOCUS_STYLE_CLASS);
        });
    }

    function setFocusIndicator(thumbnails, index) {
        clearFocusIndicators(thumbnails);
        if (index >= 0 && index < thumbnails.length) {
            thumbnails[index].classList.add(FOCUS_STYLE_CLASS);
        }
    }

    function navigateGallery(direction) {
        const container = getGalleryContainer();
        if (!container) return;

        const thumbnails = getThumbnails(container);
        if (thumbnails.length === 0) return;

        const totalImages = thumbnails.length;

        if (focusedIndex < 0) {
            // No prior selection — start at first image
            focusedIndex = 0;
        } else if (direction === "right") {
            // Clamp: right at last stays at last
            focusedIndex = Math.min(focusedIndex + 1, totalImages - 1);
        } else if (direction === "left") {
            // Clamp: left at first stays at first
            focusedIndex = Math.max(focusedIndex - 1, 0);
        }

        setFocusIndicator(thumbnails, focusedIndex);

        // Click the thumbnail to trigger Gradio's selection mechanism
        if (thumbnails[focusedIndex]) {
            thumbnails[focusedIndex].click();
        }
    }

    function triggerSendToAction(action) {
        // Trigger the corresponding send-to button click programmatically.
        // Buttons live inside the #gallery-actions row.
        var actionsRow = document.getElementById("gallery-actions");
        if (!actionsRow) return;

        var buttons = actionsRow.querySelectorAll("button");
        for (var i = 0; i < buttons.length; i++) {
            var btnText = (buttons[i].textContent || "").trim().toLowerCase();
            if (action === "img2img" && btnText.indexOf("img2img") !== -1) {
                buttons[i].click();
                return;
            }
            if (action === "upscale" && btnText.indexOf("upscale") !== -1) {
                buttons[i].click();
                return;
            }
        }
    }

    function initFocusOnEntry(container) {
        const thumbnails = getThumbnails(container);
        if (thumbnails.length === 0) return;

        if (focusedIndex < 0) {
            // Set focus to first image when gallery receives focus with
            // no prior selection (Requirement 7.6)
            focusedIndex = 0;
            setFocusIndicator(thumbnails, focusedIndex);
            if (thumbnails[focusedIndex]) {
                thumbnails[focusedIndex].click();
            }
        } else {
            // Restore visual indicator on previously focused image
            setFocusIndicator(thumbnails, focusedIndex);
        }
    }

    function setupGallery() {
        const container = getGalleryContainer();
        if (!container) {
            // Gallery not yet in DOM — retry later
            setTimeout(setupGallery, 500);
            return;
        }

        // Make the gallery container focusable
        if (!container.getAttribute("tabindex")) {
            container.setAttribute("tabindex", "0");
        }

        // Listen for keydown events on the gallery
        container.addEventListener("keydown", function(event) {
            if (event.key === "ArrowRight") {
                event.preventDefault();
                navigateGallery("right");
            } else if (event.key === "ArrowLeft") {
                event.preventDefault();
                navigateGallery("left");
            } else if (event.key === "i" && focusedIndex >= 0) {
                // Keyboard shortcut: 'i' sends focused image to img2img
                event.preventDefault();
                triggerSendToAction("img2img");
            } else if (event.key === "u" && focusedIndex >= 0) {
                // Keyboard shortcut: 'u' sends focused image to upscale
                event.preventDefault();
                triggerSendToAction("upscale");
            }
        });

        // When gallery receives focus (tab or click), initialize focus
        container.addEventListener("focus", function() {
            initFocusOnEntry(container);
        }, true);

        // Also listen for clicks on thumbnails to sync focusedIndex
        container.addEventListener("click", function(event) {
            const thumbnails = getThumbnails(container);
            if (thumbnails.length === 0) return;

            // Find which thumbnail was clicked
            for (let i = 0; i < thumbnails.length; i++) {
                if (thumbnails[i].contains(event.target) || thumbnails[i] === event.target) {
                    focusedIndex = i;
                    setFocusIndicator(thumbnails, focusedIndex);
                    break;
                }
            }
        });

        // Watch for gallery content changes (new images loaded)
        // Reset focus index if images change
        const observer = new MutationObserver(function() {
            const thumbnails = getThumbnails(container);
            if (thumbnails.length === 0) {
                focusedIndex = -1;
                return;
            }
            // Clamp focusedIndex if gallery shrunk
            if (focusedIndex >= thumbnails.length) {
                focusedIndex = thumbnails.length - 1;
            }
            // Re-apply visual indicator if still valid
            if (focusedIndex >= 0) {
                setFocusIndicator(thumbnails, focusedIndex);
            }
        });

        observer.observe(container, { childList: true, subtree: true });
    }

    // Initialize when DOM is ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", setupGallery);
    } else {
        // DOM already ready, but Gradio may not have rendered yet
        setTimeout(setupGallery, 300);
    }
})();
</script>
"""


def on_gallery_select(evt: gr.SelectData) -> str | None:
    """Update selected image state when user selects a gallery image.

    This function is intended to be wired to ``gallery.select()`` inside
    a ``gr.Blocks`` context (done in the integration task that assembles
    app.py). It extracts the file path from the Gradio SelectData event.

    Args:
        evt: Gradio SelectData event containing the selected image info.

    Returns:
        The file path of the selected image, or None if unavailable.
    """
    if evt is not None and evt.value is not None:
        value = evt.value
        if isinstance(value, dict):
            # Gradio may nest the path under "image" -> "path" or "name"
            return value.get("image", {}).get("path") or value.get("name")
        if isinstance(value, str):
            return value
    return None


def create_gallery_with_actions() -> tuple[gr.Gallery, gr.Button, gr.Button, gr.State]:
    """Create a Gallery component with send-to action buttons and keyboard navigation.

    The gallery displays generation results (single-image or multi-image
    batch) and provides action buttons that appear below the gallery for
    the selected image. Keyboard navigation (ArrowLeft/ArrowRight) is
    injected as client-side JavaScript for zero-latency response.

    This function is designed to be called within a ``gr.Blocks`` context.
    When called inside Blocks, the ``gallery.select`` event is automatically
    wired to update the selected image state. When called outside Blocks
    (e.g. in tests), the components are still created but event wiring is
    skipped.

    Components returned:
    - Gallery: Displays generated images with selection support.
    - "Send to img2img" button: Sends the selected image to the img2img
      workflow (handler wired in a later task).
    - "Send to Upscale" button: Sends the selected image to the upscaler
      pipeline (handler wired in a later task).
    - Selected image state: A gr.State component that tracks which image
      is currently selected, so handlers know what to operate on.

    Returns:
        Tuple of (gallery, send_to_img2img_btn, send_to_upscale_btn,
        selected_image_state).
    """
    # Gallery for displaying generation results
    gallery = gr.Gallery(
        label="Results",
        columns=2,
        height="auto",
        object_fit="contain",
        show_download_button=True,
        show_share_button=False,
        elem_id="results-gallery",
    )

    # Hidden state to track the currently selected image path
    # Updated when the user selects/clicks an image in the gallery
    selected_image_state = gr.State(value=None)

    # Action buttons displayed below the gallery
    with gr.Row(elem_id="gallery-actions"):
        send_to_img2img_btn = gr.Button(
            value="Send to img2img",
            variant="secondary",
            size="sm",
            interactive=True,
        )
        send_to_upscale_btn = gr.Button(
            value="Send to Upscale",
            variant="secondary",
            size="sm",
            interactive=True,
        )

    # Inject keyboard navigation JavaScript (Requirements 7.1–7.7).
    # The script runs client-side for zero-latency arrow key response.
    gr.HTML(GALLERY_KEYBOARD_JS, visible=False)

    # Wire gallery selection to update the selected image state.
    # This requires a gr.Blocks context; if called outside one (e.g. in
    # unit tests), skip the wiring gracefully.
    try:
        gallery.select(
            fn=on_gallery_select,
            inputs=None,
            outputs=[selected_image_state],
        )
    except AttributeError:
        # Called outside gr.Blocks context — skip event wiring.
        # The caller (app.py) will wire events in the integration task.
        pass

    return gallery, send_to_img2img_btn, send_to_upscale_btn, selected_image_state
