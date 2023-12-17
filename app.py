import os  # added for cache_examples
from pathlib import Path

import gradio as gr
import numpy as np
import supervision as sv
from gradio import ColorPicker
from PIL import Image
from torch import cuda, device
from ultralytics import YOLO

from src.gradio_desc import BANNER, DESC, SUBTITLE, TITLE
from src.sam_tab import sam_annotator_tab
from src.video_tab import video_annotator_tab

# Use GPU if available
if cuda.is_available():
    device = device("cuda")
else:
    device = device("cpu")


last_detections = sv.Detections.empty()
last_labels: list[str] = []


def load_model(img, model: str | Path = "yolov8s-seg.pt"):
    # Load model, get results and return detections/labels
    model = YOLO(model=model)
    result = model(img, verbose=False, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

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


def annotators(
    img,
    last_detections,
    annotators_list,
    last_labels,
    colorbb,
    colormask,
    colorellipse,
    colorbc,
    colorcir,
    colorlabel,
    colorhalo,
    colortri,
    colordot,
) -> np.ndarray:
    if last_detections == sv.Detections.empty():
        gr.Warning("Detection is empty please add image and annotate first")
        return np.zeros()

    if "Blur" in annotators_list:
        # Apply Blur
        blur_annotator = sv.BlurAnnotator()
        img = blur_annotator.annotate(img, detections=last_detections)

    if "BoundingBox" in annotators_list:
        # Draw Boundingbox
        box_annotator = sv.BoundingBoxAnnotator(sv.Color.from_hex(str(colorbb)))
        img = box_annotator.annotate(img, detections=last_detections)

    if "Mask" in annotators_list:
        # Draw Mask
        mask_annotator = sv.MaskAnnotator(sv.Color.from_hex(str(colormask)))
        img = mask_annotator.annotate(img, detections=last_detections)

    if "Ellipse" in annotators_list:
        # Draw Ellipse
        ellipse_annotator = sv.EllipseAnnotator(sv.Color.from_hex(str(colorellipse)))
        img = ellipse_annotator.annotate(img, detections=last_detections)

    if "BoxCorner" in annotators_list:
        # Draw Box corner
        corner_annotator = sv.BoxCornerAnnotator(sv.Color.from_hex(str(colorbc)))
        img = corner_annotator.annotate(img, detections=last_detections)

    if "Circle" in annotators_list:
        # Draw Circle
        circle_annotator = sv.CircleAnnotator(sv.Color.from_hex(str(colorcir)))
        img = circle_annotator.annotate(img, detections=last_detections)

    if "Label" in annotators_list:
        # Draw Label
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
        label_annotator = sv.LabelAnnotator(sv.Color.from_hex(str(colorlabel)))
        img = label_annotator.annotate(
            img, detections=last_detections, labels=last_labels
        )

    if "Pixelate" in annotators_list:
        # Apply PixelateAnnotator
        pixelate_annotator = sv.PixelateAnnotator()
        img = pixelate_annotator.annotate(img, detections=last_detections)

    if "Halo" in annotators_list:
        # Draw HaloAnnotator
        halo_annotator = sv.HaloAnnotator(sv.Color.from_hex(str(colorhalo)))
        img = halo_annotator.annotate(img, detections=last_detections)

    if "HeatMap" in annotators_list:
        # Draw HeatMapAnnotator
        heatmap_annotator = sv.HeatMapAnnotator()
        img = heatmap_annotator.annotate(img, detections=last_detections)

    if "Dot" in annotators_list:
        # Dot DotAnnotator
        dot_annotator = sv.DotAnnotator(sv.Color.from_hex(str(colordot)))
        img = dot_annotator.annotate(img, detections=last_detections)

    if "Triangle" in annotators_list:
        # Draw TriangleAnnotator
        tri_annotator = sv.TriangleAnnotator(sv.Color.from_hex(str(colortri)))
        img = tri_annotator.annotate(img, detections=last_detections)

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


def annotator(
    img,
    model,
    annotators_list,
    colorbb,
    colormask,
    colorellipse,
    colorbc,
    colorcir,
    colorlabel,
    colorhalo,
    colortri,
    colordot,
    progress=gr.Progress(track_tqdm=True),
) -> np.ndarray:
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

    detections, labels = load_model(img, model)
    last_detections = detections
    last_labels = labels

    return annotators(
        img,
        last_detections,
        annotators_list,
        last_labels,
        colorbb,
        colormask,
        colorellipse,
        colorbc,
        colorcir,
        colorlabel,
        colorhalo,
        colortri,
        colordot,
    )


purple_theme = theme = gr.themes.Soft(primary_hue=gr.themes.colors.purple).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    checkbox_label_background_fill_selected="*primary_600",
    checkbox_background_color_selected="*primary_400",
)

with gr.Blocks(theme=purple_theme) as app:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(BANNER)
    gr.HTML(DESC)

    with gr.Tab("Image Annotator(YoloV8)"):
        models = gr.Dropdown(
            [
                "yolov8n-seg.pt",
                "yolov8s-seg.pt",
                "yolov8m-seg.pt",
                "yolov8l-seg.pt",
                "yolov8x-seg.pt",
            ],
            type="value",
            value="yolov8s-seg.pt",
            label="Select Model:",
        )

        annotators_list = gr.CheckboxGroup(
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
                "Triangle",
            ],
            value=["BoundingBox", "Mask"],
            label="Select Annotators:",
        )

        gr.Markdown("## Color Picker 🎨")
        with gr.Row(variant="panel"):
            with gr.Column():
                colorbb = gr.ColorPicker(value="#A351FB", label="BoundingBox")
                colormask = gr.ColorPicker(value="#A351FB", label="Mask")
                colorellipse = gr.ColorPicker(value="#A351FB", label="Ellipse")
            with gr.Column():
                colorbc = gr.ColorPicker(value="#A351FB", label="BoxCorner")
                colorcir = gr.ColorPicker(value="#A351FB", label="Circle")
                colorlabel = gr.ColorPicker(value="#A351FB", label="Label")
            with gr.Column():
                colorhalo = gr.ColorPicker(value="#A351FB", label="Halo")
                colordot = gr.ColorPicker(value="#A351FB", label="Dot")
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
                models,
                annotators_list,
                colorbb,
                colormask,
                colorellipse,
                colorbc,
                colorcir,
                colorlabel,
                colorhalo,
                colortri,
                colordot,
            ],
            outputs=image_output,
        )

        gr.Markdown("## Image Examples 🖼️")
        gr.Examples(
            examples=[
                os.path.join(os.path.abspath(""), "./assets/city.jpg"),
                os.path.join(os.path.abspath(""), "./assets/household.jpg"),
                os.path.join(os.path.abspath(""), "./assets/industry.jpg"),
                os.path.join(os.path.abspath(""), "./assets/retail.jpg"),
                os.path.join(os.path.abspath(""), "./assets/aerodefence.jpg"),
            ],
            inputs=image_input,
            outputs=image_output,
            fn=annotator,
            cache_examples=False,
        )

        annotators_list.change(
            fn=annotator,
            inputs=[
                image_input,
                models,
                annotators_list,
                colorbb,
                colormask,
                colorellipse,
                colorbc,
                colorcir,
                colorlabel,
                colorhalo,
                colortri,
                colordot,
            ],
            outputs=image_output,
        )

        def change_color(color: ColorPicker):
            color.change(
                fn=annotator,
                inputs=[
                    image_input,
                    models,
                    annotators_list,
                    colorbb,
                    colormask,
                    colorellipse,
                    colorbc,
                    colorcir,
                    colorlabel,
                    colorhalo,
                    colortri,
                    colordot,
                ],
                outputs=image_output,
            )

        colors = [
            colorbb,
            colormask,
            colorellipse,
            colorbc,
            colorcir,
            colorlabel,
            colorhalo,
            colortri,
            colordot,
        ]

        for color in colors:
            change_color(color)

    sam_annotator_tab(gr)
    video_annotator_tab(gr)


if __name__ == "__main__":
    print("Starting app...")
    print("Dark theme is available at: http://localhost:7860/?__theme=dark")
    app.launch(debug=False, server_name="0.0.0.0")  # for local network
    # app.launch(debug=False)
