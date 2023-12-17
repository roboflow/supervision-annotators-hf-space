import gradio as gr

from src.sam.sam_examples import BOX_EXAMPLES, POINT_EXAMPLES
from src.sam.sam_inference import SamInference

# gradio Segmentation Anything Demo

sam_inference = SamInference()


def sam_segmentation(model, progress=gr.Progress(track_tqdm=True)):
    sam_inference.set_sam_model(model, progress=gr.Progress(track_tqdm=True))
    sam_inference.set_sam_processor(model, progress=gr.Progress(track_tqdm=True))
    return "Model is ready"


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
download_status_label = gr.Label("Model is ready", show_label=False)


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
        gr.Examples(
            fn=sam_inference.sam_box_inference,
            examples=BOX_EXAMPLES,
            inputs=[
                box_input_image,
                x_min_number,
                y_min_number,
                x_max_number,
                y_max_number,
            ],
            outputs=[box_output_image],
        )
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
        gr.Examples(
            fn=sam_inference.sam_point_inference,
            examples=POINT_EXAMPLES,
            inputs=[point_input_image, x_number, y_number],
            outputs=[point_output_image],
        )


def sam_models(gr: gr):
    with gr.Row():
        models = gr.Dropdown(
            [
                "facebook/sam-vit-base",
                "facebook/sam-vit-large",
                "facebook/sam-vit-huge",
            ],
            type="value",
            value="facebook/sam-vit-huge",
            label="Select SAM Model:",
            min_width=900,
        )
        download_status_label.render()
        return models


def sam_annotator_tab(gr: gr):
    """Run the SAM menu."""
    with gr.Tab("Image Annotator(SAM)"):
        model = sam_models(gr)
        box_and_point_menu(gr)
        model.select(
            fn=sam_segmentation, inputs=[model], outputs=[download_status_label]
        )

        point_annotate_button.click(
            fn=sam_inference.sam_point_inference,
            inputs=[point_input_image, x_number, y_number],
            outputs=[point_output_image],
        )

        box_annotate_button.click(
            fn=sam_inference.sam_box_inference,
            inputs=[
                box_input_image,
                x_min_number,
                y_min_number,
                x_max_number,
                y_max_number,
            ],
            outputs=[box_output_image],
        )
