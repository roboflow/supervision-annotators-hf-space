from pathlib import Path

import gradio as gr
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO

from src import DEVICE
from src.utils import calculate_crop_dim

device = DEVICE
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
