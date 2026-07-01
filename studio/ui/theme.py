"""Glassmorphism theme for Cinderworks Studio.

Animated multi-stop gradient background, frosted glass panels,
lemon/amber accent palette. Sourced from BeatBunny CUSTOM_CSS block
with accent colors adapted for Cinderworks identity.
"""

try:
    import gradio as gr
except ImportError:
    gr = None

# --- Glassmorphism CSS ---
CUSTOM_CSS = """
/* Background: Deep animated multi-stop gradient */
body, .gradio-container {
    background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white !important;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism Panels */
.glass-panel {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Forced light text for contrast on dark/gradient backgrounds */
h1, h2, h3 {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

p, span, label, .prose {
    color: #eee !important;
}

/* Inputs & Textboxes */
textarea, input {
    background-color: rgba(0, 0, 0, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #eee !important;
}

/* Lemon/Amber accent buttons */
button.primary {
    background: linear-gradient(90deg, #f5af19, #f12711, #f5af19) !important;
    background-size: 200% 200% !important;
    animation: btn-shimmer 3s ease infinite !important;
    border: none !important;
    color: white !important;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    box-shadow: 0 0 15px rgba(245, 175, 25, 0.4) !important;
    transition: all 0.3s ease;
}

button.primary:hover {
    box-shadow: 0 0 25px rgba(245, 175, 25, 0.7) !important;
    transform: translateY(-2px);
}

@keyframes btn-shimmer {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Secondary buttons */
button.secondary {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(245, 175, 25, 0.3) !important;
    color: #f5d020 !important;
    transition: all 0.3s ease;
}

button.secondary:hover {
    background: rgba(245, 175, 25, 0.15) !important;
    border-color: rgba(245, 175, 25, 0.6) !important;
}

/* Amber accent for links, highlights, and interactive elements */
a, .accent {
    color: #f5d020 !important;
}

/* Tab styling */
.tab-nav button {
    color: rgba(255, 255, 255, 0.7) !important;
    border: none !important;
    transition: color 0.2s ease;
}

.tab-nav button.selected {
    color: #f5d020 !important;
    border-bottom: 2px solid #f5af19 !important;
}

/* Sliders and range inputs — amber accent */
input[type="range"]::-webkit-slider-thumb {
    background: #f5af19 !important;
}

input[type="range"]::-moz-range-thumb {
    background: #f5af19 !important;
}

/* Progress bars */
.progress-bar {
    background: linear-gradient(90deg, #f5af19, #f5d020) !important;
}

/* Readiness banner styling */
#readiness-banner {
    background: rgba(245, 175, 25, 0.1) !important;
    border: 1px solid rgba(245, 175, 25, 0.3) !important;
    border-radius: 12px !important;
    color: #f5d020 !important;
}

/* Gallery/image display */
.gallery-item {
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
}

::-webkit-scrollbar-thumb {
    background: rgba(245, 175, 25, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(245, 175, 25, 0.6);
}
"""


def get_theme():
    """Return a Gradio theme configured for the glassmorphism aesthetic.

    Uses Gradio Soft theme with amber/yellow hues and transparent backgrounds
    so the animated CSS gradient shows through. The CUSTOM_CSS string should
    be passed separately to gr.Blocks(css=CUSTOM_CSS).

    Returns None if gradio is not available (e.g. during testing).
    """
    if gr is None:
        return None

    return gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="transparent",
        block_background_fill="transparent",
        block_border_width="0px",
    )
