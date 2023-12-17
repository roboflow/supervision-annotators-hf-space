import gradio as gr
import supervision as sv
import torch
from transformers import SamModel, SamProcessor

# gradio Segmentation Anything Demo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MASK_ANNOTATOR = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)


# SAM Mode enum for point and box input and defualt to point
class SamMode:
    POINT = "Point"
    BOX = "Box"


# SAM Mode enum default to point
sam_mode: SamMode = SamMode.POINT


def sam_segmentation(input, model: str):
    """Run the segmentation model on the input image."""
    SamModel.from_pretrained(model).to(DEVICE)
    SamProcessor.from_pretrained(model)


box_input_image = gr.Image(label="SAM Box Input")
box_output_image = gr.Image(label="SAM Box Output")
point_input_image = gr.Image(label="SAM Point Input")
point_output_image = gr.Image(label="SAM Point Output")
x_min_number = gr.Number(label="x_min")
y_min_number = gr.Number(label="y_min")
x_max_number = gr.Number(label="x_max")
y_max_number = gr.Number(label="y_max")
x_number = gr.Number(label="x")
y_number = gr.Number(label="y")
box_annotate_button = gr.Button(value="Annotate it!", variant="primary")
point_annotate_button = gr.Button(value="Annotate it!", variant="primary")


def box_and_point_menu(gr: gr):
    """Return a dropdown menu for the box and point annotator."""
    with gr.TabItem("Box Selection"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        box_input_image.render()
                    with gr.Column():
                        box_output_image.render()
                        box_annotate_button.render()
                with gr.Accordion(label="Box", open=True):
                    with gr.Row():
                        x_min_number.render()
                        y_min_number.render()
                        x_max_number.render()
                        y_max_number.render()
    with gr.TabItem("Point Selection"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        point_input_image.render()
                    with gr.Column():
                        point_output_image.render()
                        point_annotate_button.render()
                with gr.Accordion(label="Point", open=True):
                    with gr.Row():
                        x_number.render()
                        y_number.render()


def sam_models(gr: gr):
    models = gr.Dropdown(
        [
            "facebook/sam-vit-base",
            "facebook/sam-vit-large",
            "facebook/sam-vit-huge",
        ],
        type="value",
        value="facebook/sam-vit-huge",
        label="Select SAM Model:",
    )
    return models


def sam_annotator_tab(gr: gr):
    """Run the SAM menu."""
    with gr.Tab("Image Annotator(SAM)"):
        model = sam_models(gr)
        box_and_point_menu(gr)
        box_annotate_button.click(
            fn=sam_segmentation, inputs=[model, model], outputs=[box_output_image]
        )
        point_annotate_button.click(
            fn=sam_segmentation, inputs=[model, model], outputs=[point_output_image]
        )
