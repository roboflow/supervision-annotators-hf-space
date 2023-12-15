import os  # added for cache_examples

import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from torch import cuda, device
from ultralytics import YOLO

# Use GPU if available
if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")


TITLE = """<h1 align="center">Supervision Annotator Playground 🚀</h1>"""
SUBTITLE = """<h2 align="center">Experiment with Supervision Annotators</h2>"""
DUPLICATE = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <a href="https://huggingface.co/spaces/Roboflow/Annotators?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space" style="margin-right: 10px;">
    </a>
</div>
"""

def load_model(img):
    # Load model, get results and return detections/labels
    model = YOLO("yolov8s-seg.pt")
    result = model(img, verbose=False, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    print(labels)
    return detections, labels


def calculate_crop_dim(a, b):
    # Calculates the crop dimensions of the image resultant
    if a > b:
        width = a
        height = a
    else:
        width = b
        height = b

    return width, height


def annotator(
    img,
    annotators,
    colorbb,
    colormask,
    colorellipse,
    colorbc,
    colorcir,
    colorlabel,
    colorhalo,
    colortri
):
    """
    Function that changes the color of annotators
    Args:
        annotators: Icon whose color needs to be changed.
        color: Chosen color with which to edit the input icon in Hex.
        img: Input image is numpy matrix in BGR.
    Returns:
        annotators: annotated image
    """

    img = img[..., ::-1].copy()  # BGR to RGB using numpy

    detections, labels = load_model(img)

    if "Blur" in annotators:
        # Apply Blur
        blur_annotator = sv.BlurAnnotator()
        img = blur_annotator.annotate(img, detections=detections)

    if "BoundingBox" in annotators:
        # Draw Bounding box
        box_annotator = sv.BoundingBoxAnnotator(sv.Color.from_hex(str(colorbb)))
        img = box_annotator.annotate(img, detections=detections)

    if "Mask" in annotators:
        # Draw Mask
        mask_annotator = sv.MaskAnnotator(sv.Color.from_hex(str(colormask)))
        img = mask_annotator.annotate(img, detections=detections)

    if "Ellipse" in annotators:
        # Draw ellipse
        ellipse_annotator = sv.EllipseAnnotator(sv.Color.from_hex(str(colorellipse)))
        img = ellipse_annotator.annotate(img, detections=detections)

    if "BoxCorner" in annotators:
        # Draw Box corner
        corner_annotator = sv.BoxCornerAnnotator(sv.Color.from_hex(str(colorbc)))
        img = corner_annotator.annotate(img, detections=detections)

    if "Circle" in annotators:
        # Draw circle
        circle_annotator = sv.CircleAnnotator(sv.Color.from_hex(str(colorcir)))
        img = circle_annotator.annotate(img, detections=detections)

    if "Label" in annotators:
        # Draw Label
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        label_annotator = sv.LabelAnnotator(sv.Color.from_hex(str(colorlabel)))
        img = label_annotator.annotate(img, detections=detections, labels=labels)

    if "Pixelate" in annotators:
        # Draw PixelateAnnotator
        pixelate_annotator = sv.PixelateAnnotator()
        img = pixelate_annotator.annotate(img, detections=detections)

    if "Halo" in annotators:
        # Draw HaloAnnotator
        halo_annotator = sv.HaloAnnotator(sv.Color.from_hex(str(colorhalo)))
        img = halo_annotator.annotate(img, detections=detections)

    if "HeatMap" in annotators:
        # Draw HeatMapAnnotator
        heatmap_annotator = sv.HeatMapAnnotator()
        img = heatmap_annotator.annotate(img, detections=detections)
    
    if "Dot" in annotators:
        # Draw DotAnnotator
        dot_annotator = sv.DotAnnotator()
        img = dot_annotator.annotate(img, detections=detections)
    
    if "Triangle" in annotators:
        # Draw TriangleAnnotator
        tri_annotator = sv.TriangleAnnotator(sv.Color.from_hex(str(colortri)))
        img = tri_annotator.annotate(img, detections=detections)

    # crop image for the largest possible square
    res_img = Image.fromarray(img)
    # print(type(res_img))
    x = 0
    y = 0

    # print("size of the pil im=", res_img.size)
    (v1, v2) = res_img.size
    width, height = calculate_crop_dim(v1, v2)
    # print(width, height)
    my_img = np.array(res_img)

    crop_img = my_img[y : y + height, x : x + width]
    # print(type(crop_img))

    return crop_img[..., ::-1].copy()  # BGR to RGB using numpy


purple_theme = theme = gr.themes.Soft(primary_hue=gr.themes.colors.purple).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    checkbox_label_background_fill_selected="*primary_600",
    checkbox_background_color_selected="*primary_400",
)

with gr.Blocks(theme=purple_theme) as app:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DUPLICATE)
    annotators = gr.CheckboxGroup(
        choices=[
            "BoundingBox",
            "Mask",
            "Halo",
            "Ellipse",
            "BoxCorner",
            "Circle",
            "Label",
            "Blur",
            "Pixelate",
            "HeatMap",
            "Dot",
            "Triangle"
        ],
        value=["BoundingBox", "Mask"],
        label="Select Annotators:",
    )

    gr.Markdown("🎨 **Color Picker**")
    with gr.Row(variant="compact"):
        with gr.Column():
            colorbb = gr.ColorPicker(value="#A351FB", label="BoundingBox")
        with gr.Column():
            colormask = gr.ColorPicker(value="#A351FB", label="Mask")
        with gr.Column():
            colorellipse = gr.ColorPicker(value="#A351FB", label="Ellipse")
        with gr.Column():
            colorbc = gr.ColorPicker(value="#A351FB", label="BoxCorner")
        with gr.Column():
            colorcir = gr.ColorPicker(value="#A351FB", label="Circle")
        with gr.Column():
            colorlabel = gr.ColorPicker(value="#A351FB", label="Label")
        with gr.Column():
            colorhalo = gr.ColorPicker(value="#A351FB", label="Halo")
        with gr.Column():
            colordot = gr.ColorPicker(value="#A351FB", label="Dot")
        with gr.Column():
            colortri = gr.ColorPicker(value="#A351FB", label="Triangle")

    with gr.Row():
        with gr.Column():
            with gr.Tab("Input image"):
                image_input = gr.Image(type="numpy", show_label=False)
        with gr.Column():
            with gr.Tab("Result image"):
                image_output = gr.Image(type="numpy", show_label=False)
    image_button = gr.Button(value="Annotate it!", variant="primary")

    image_button.click(
        annotator,
        inputs=[
            image_input,
            annotators,
            colorbb,
            colormask,
            colorellipse,
            colorbc,
            colorcir,
            colorlabel,
            colorhalo,
            colortri,
        ],
        outputs=image_output,
    )

    gr.Markdown("## Image Examples")
    gr.Examples(
        examples=[
            os.path.join(os.path.abspath(""), "city.jpg"),
            os.path.join(os.path.abspath(""), "household.jpg"),
            os.path.join(os.path.abspath(""), "industry.jpg"),
            os.path.join(os.path.abspath(""), "retail.jpg"),
            os.path.join(os.path.abspath(""), "aerodefence.jpg"),
        ],
        inputs=image_input,
        outputs=image_output,
        fn=annotator,
        cache_examples=False,
    )


if __name__ == "__main__":
    print("Starting app...")
    print("Dark theme is available at: http://localhost:7860/?__theme=dark")
    app.launch(debug=False)
