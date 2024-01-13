import gradio as gr
import os
from src.annotator_tab import annotator_tab
from src.gradio_desc import gradio_desc
from src.sam_tab import sam_annotator_tab

# from src.video_tab import video_annotator_tab

purple_theme = theme = gr.themes.Soft(primary_hue=gr.themes.colors.purple).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    checkbox_label_background_fill_selected="*primary_600",
    checkbox_background_color_selected="*primary_400",
)

with gr.Blocks(theme=purple_theme) as app:
    gradio_desc(gr)
    annotator_tab(gr)
    sam_annotator_tab(gr)
    # video_annotator_tab(gr)


if __name__ == "__main__":
    # Check is SPACE_REPO_NAME env var is set
    if "SPACE_REPO_NAME" in os.environ:
        print("Starting app in space repo...")
        app.launch(debug=False)

    print("Starting app in outside space repo... (Debug mode))")
    print("Dark theme is available at: http://localhost:7860/?__theme=dark")
    app.launch(debug=True, server_name="0.0.0.0")  # for local network

