import gradio as gr
import numpy as np
import supervision as sv
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from src import DEVICE
from src.utils import calculate_dynamic_circle_radius, draw_circle


# Sam Inferece Class
class SamInference:
    def __init__(self):
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(DEVICE)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
        self.prompt_color = sv.Color.from_hex("#D3D3D3")
        self.mask_color = sv.Color.from_hex("#FF0000")

    def set_sam_model(self, model: str, progress=gr.Progress(track_tqdm=True)):
        self.sam_model = SamModel.from_pretrained(model).to(DEVICE)

    def set_sam_processor(self, processor: str, progress=gr.Progress(track_tqdm=True)):
        self.sam_processor = SamProcessor.from_pretrained(processor)

    def annotate_image_with_point_prompt_result(
        self, image: np.ndarray, detections: sv.Detections, x: int, y: int
    ) -> np.ndarray:
        h, w, _ = image.shape
        bgr_image = image[:, :, ::-1]
        annotated_bgr_image = self.mask_annotator.annotate(
            scene=bgr_image, detections=detections
        )
        annotated_bgr_image = draw_circle(
            scene=annotated_bgr_image,
            center=sv.Point(x=x, y=y),
            radius=calculate_dynamic_circle_radius(resolution_wh=(w, h)),
            color=self.prompt_color,
        )
        return annotated_bgr_image[:, :, ::-1]

    def sam_point_inference(self, image: np.ndarray, x: int, y: int) -> np.ndarray:
        input_points = [[[x, y]]]
        inputs = self.sam_processor(
            Image.fromarray(image), input_points=[input_points], return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        mask = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0][0].numpy()
        mask = mask[np.newaxis, ...]
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
        return self.annotate_image_with_point_prompt_result(
            image=image, detections=detections, x=x, y=y
        )

    def annotate_image_with_box_prompt_result(
        self,
        image: np.ndarray,
        detections: sv.Detections,
        x_min: int,
        y_min: int,
        x_max: int,
        y_max: int,
    ) -> np.ndarray:
        h, w, _ = image.shape
        bgr_image = image[:, :, ::-1]
        annotated_bgr_image = self.mask_annotator.annotate(
            scene=bgr_image, detections=detections
        )
        annotated_bgr_image = sv.draw_rectangle(
            scene=annotated_bgr_image,
            rect=sv.Rect(
                x=x_min,
                y=y_min,
                width=int(x_max - x_min),
                height=int(y_max - y_min),
            ),
            color=self.prompt_color,
            thickness=sv.calculate_dynamic_line_thickness(resolution_wh=(w, h)),
        )
        return annotated_bgr_image[:, :, ::-1]

    def sam_box_inference(
        self, image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int
    ) -> np.ndarray:
        input_boxes = [[[x_min, y_min, x_max, y_max]]]
        inputs = self.sam_processor(
            Image.fromarray(image), input_boxes=[input_boxes], return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        mask = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0][0].numpy()
        mask = mask[np.newaxis, ...]
        detections = sv.Detections(xyxy=sv.mask_to_xyxy(masks=mask), mask=mask)
        return self.annotate_image_with_box_prompt_result(
            image=image,
            detections=detections,
            x_max=x_max,
            x_min=x_min,
            y_max=y_max,
            y_min=y_min,
        )
