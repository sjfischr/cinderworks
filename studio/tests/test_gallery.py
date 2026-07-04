"""Unit tests for ui/gallery.py — Gallery component with send-to actions.

Validates Requirements 9.1, 9.2, 9.3, 9.4, 7.7:
- Gallery component is created with correct configuration
- Send-to buttons exist and have correct labels
- Selected image state is initialized to None
- Gallery supports both single-image and multi-image results
- Keyboard shortcuts for send-to actions are present in injected JS
"""

import gradio as gr

from studio.ui.gallery import create_gallery_with_actions, GALLERY_KEYBOARD_JS


class TestCreateGalleryWithActions:
    """Tests for create_gallery_with_actions() component creation."""

    def test_returns_four_components(self):
        """Function returns a tuple of four components."""
        result = create_gallery_with_actions()
        assert len(result) == 4

    def test_gallery_is_gradio_gallery(self):
        """First component is a Gradio Gallery."""
        gallery, _, _, _ = create_gallery_with_actions()
        assert isinstance(gallery, gr.Gallery)

    def test_gallery_has_results_label(self):
        """Gallery has the 'Results' label."""
        gallery, _, _, _ = create_gallery_with_actions()
        assert gallery.label == "Results"

    def test_gallery_elem_id(self):
        """Gallery has the expected elem_id for CSS/JS targeting."""
        gallery, _, _, _ = create_gallery_with_actions()
        assert gallery.elem_id == "results-gallery"

    def test_send_to_img2img_button_exists(self):
        """Send to img2img button is a Gradio Button with correct label."""
        _, send_img2img, _, _ = create_gallery_with_actions()
        assert isinstance(send_img2img, gr.Button)
        assert send_img2img.value == "Send to img2img"

    def test_send_to_upscale_button_exists(self):
        """Send to Upscale button is a Gradio Button with correct label."""
        _, _, send_upscale, _ = create_gallery_with_actions()
        assert isinstance(send_upscale, gr.Button)
        assert send_upscale.value == "Send to Upscale"

    def test_selected_image_state_initialized_none(self):
        """Selected image state is a gr.State initialized to None."""
        _, _, _, selected_state = create_gallery_with_actions()
        assert isinstance(selected_state, gr.State)
        assert selected_state.value is None

    def test_buttons_are_interactive(self):
        """Both action buttons are interactive (clickable)."""
        _, send_img2img, send_upscale, _ = create_gallery_with_actions()
        assert send_img2img.interactive is True
        assert send_upscale.interactive is True

    def test_gallery_columns_for_batch_display(self):
        """Gallery uses multi-column layout for batch results."""
        gallery, _, _, _ = create_gallery_with_actions()
        # columns=2 supports multi-image batch display
        assert gallery.columns == 2


class TestGalleryKeyboardShortcuts:
    """Tests that keyboard shortcuts for send-to actions are wired in gallery JS.

    Validates Requirement 9.4: send-to actions accessible via keyboard shortcut
    while image is focused in Gallery.
    """

    def test_keyboard_js_contains_img2img_shortcut(self):
        """GALLERY_KEYBOARD_JS includes 'i' key shortcut for send to img2img."""
        assert '"i"' in GALLERY_KEYBOARD_JS
        assert "img2img" in GALLERY_KEYBOARD_JS

    def test_keyboard_js_contains_upscale_shortcut(self):
        """GALLERY_KEYBOARD_JS includes 'u' key shortcut for send to upscale."""
        assert '"u"' in GALLERY_KEYBOARD_JS
        assert "upscale" in GALLERY_KEYBOARD_JS

    def test_shortcuts_require_focused_image(self):
        """Shortcuts are only active when focusedIndex >= 0 (image focused)."""
        assert "focusedIndex >= 0" in GALLERY_KEYBOARD_JS

    def test_trigger_send_to_action_function_exists(self):
        """The triggerSendToAction helper function is defined in the JS."""
        assert "triggerSendToAction" in GALLERY_KEYBOARD_JS

    def test_shortcuts_target_gallery_actions_row(self):
        """Shortcuts find buttons via the gallery-actions elem_id."""
        assert "gallery-actions" in GALLERY_KEYBOARD_JS
