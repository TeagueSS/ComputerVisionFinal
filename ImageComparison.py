import cv2
import numpy as np
import os
import tkinter as tk  # Corrected: Main tkinter import
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json  # Moved to top
import torch  # Import torch to check MPS availability

from ultralytics import YOLO
from ultralytics.engine.results import Results  # Moved to top
from typing import List, Optional  # Moved to top, using Optional


# --- Helper Functions ---

def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Loads an image from the specified file path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[np.ndarray]: The loaded image as a NumPy array (BGR format),
                              or None if the file doesn't exist or is invalid.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image (invalid format or file): {image_path}")
            return None
        print(f"Image loaded successfully from: {image_path}")
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_model(model_path: str, device: str = 'auto') -> Optional[YOLO]:
    """
    Loads a pre-trained Ultralytics YOLO model. Handles automatic download.

    Args:
        model_path (str): Path or name of the YOLO model file (e.g., 'yolov8n-seg.pt').
        device (str): Device to load the model onto ('cpu', 'cuda:0', 'mps', 'auto').
                      Defaults to 'auto'.

    Returns:
        Optional[YOLO]: The loaded YOLO model object, or None if loading fails.
    """
    resolved_device = device
    if device == 'auto':
        if torch.cuda.is_available():
            resolved_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():  # Check for MPS
            resolved_device = 'mps'
        else:
            resolved_device = 'cpu'
        print(f"Auto-detected device: {resolved_device}")
    elif device == 'mps' and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
        print("Warning: MPS device requested but not available. Falling back to CPU.")
        resolved_device = 'cpu'
    elif 'cuda' in device and not torch.cuda.is_available():
        print(f"Warning: CUDA device '{device}' requested but not available. Falling back to CPU.")
        resolved_device = 'cpu'

    try:
        model = YOLO(model_path)
        # Store the resolved device preference for later use in inference
        # Ultralytics models usually handle device internally or on predict,
        # but setting an attribute like this can be a way to pass it if needed.
        # Better to pass 'device' directly to predict.
        model.predict_device = resolved_device  # Custom attribute to store preference
        print(f"Model '{model_path}' loaded successfully. Will attempt inference on '{resolved_device}'.")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def run_inference(model: YOLO, image: np.ndarray,
                  confidence_threshold: float = 0.25,
                  iou_threshold: float = 0.7) -> Optional[Results]:
    """
    Runs object detection/segmentation inference on the image.

    Args:
        model (YOLO): The loaded Ultralytics YOLO model.
        image (np.ndarray): The input image (BGR format).
        confidence_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        Optional[Results]: The Ultralytics Results object for the single image,
                           or None on failure.
    """
    if model is None or image is None:
        print("Error: Model or image is not loaded for inference.")
        return None
    try:
        # Use the device preference stored during model loading or default to 'cpu'
        device_to_use = getattr(model, 'predict_device', 'cpu')

        # Perform inference
        # Ultralytics predict method returns a list of Results objects.
        # For a single image, this list will contain one Results object.
        results_list = model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=device_to_use,
            verbose=False  # Suppress console output from predict
        )
        if results_list and len(results_list) > 0:
            results_obj = results_list[0]  # Get the first (and only) Results object
            num_detections = len(results_obj.boxes) if results_obj.boxes is not None else 0
            print(f"Inference completed. Found {num_detections} potential objects.")
            return results_obj
        else:
            print("Inference returned no results or an empty list.")
            return None
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def visualize_results(image: np.ndarray, results_obj: Optional[Results]) -> np.ndarray:
    """
    Draws bounding boxes and/or masks from the results_obj onto the image.

    Args:
        image (np.ndarray): The original image (BGR format).
        results_obj (Optional[Results]): The Results object from inference.

    Returns:
        np.ndarray: A copy of the image with visualizations.
                    Returns a copy of the original image if results_obj is None or on error.
    """
    if image is None:  # Should not happen if logic is correct, but good check
        print("Error: No image provided for visualization.")
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Return a placeholder or raise error

    if results_obj is None:
        print("No results to visualize. Returning original image copy.")
        return image.copy()

    try:
        # Ultralytics Results object has a built-in plot method
        annotated_image = results_obj.plot()  # Returns a NumPy array (BGR)
        print("Visualization generated.")
        return annotated_image
    except Exception as e:
        print(f"Error during visualization: {e}")
        # Fallback: Return a copy of the original image on error
        return image.copy()


def save_annotations(results_obj: Optional[Results], output_path: str, image_shape: tuple) -> bool:
    """
    Saves the detection/segmentation results to a JSON file.

    Args:
        results_obj (Optional[Results]): The Results object from inference.
        output_path (str): Path to save the JSON annotation file.
        image_shape (tuple): Shape of the original image (height, width, channels).

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if not results_obj:
        print("Warning: No results object to save.")
        return False

    try:
        height, width, _ = image_shape
        annotations_list = []  # CORRECTED: Initialize as list

        # Access data from the Results object
        boxes_data = results_obj.boxes
        masks_data = results_obj.masks
        names = results_obj.names  # Class names mapping {id: name}

        if boxes_data is None or boxes_data.xyxyn is None or boxes_data.xyxy is None:
            print(
                "Warning: Essential box data (xyxyn or xyxy) not found in results. Cannot save annotations accurately.")
            return False

        # Normalized bounding boxes [x1, y1, x2, y2]
        norm_boxes_xyxyn = boxes_data.xyxyn.cpu().numpy()
        # Pixel bounding boxes [x1, y1, x2, y2]
        pixel_boxes_xyxy = boxes_data.xyxy.cpu().numpy()
        confs = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy().astype(int)

        for i in range(len(norm_boxes_xyxyn)):
            class_id = int(class_ids[i])
            class_name = names.get(class_id, f"class_{class_id}") if names else f"class_{class_id}"

            annotation = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "bbox_normalized_xyxyn": norm_boxes_xyxyn[i].tolist(),
                "bbox_pixels_xyxy": pixel_boxes_xyxy[i].astype(int).tolist()  # Using xyxy pixel coordinates
            }

            # Include mask data if available
            if masks_data is not None and masks_data.xy is not None and len(masks_data.xy) > i:
                annotation["segmentation_normalized_polygons"] = masks_data.xy[i].tolist()

            annotations_list.append(annotation)

        output_data = {
            "image_height": height,
            "image_width": width,
            "annotations": annotations_list
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Annotations saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving annotations to {output_path}: {e}")
        return False


# --- GUI Class ---
class AnnotationApp:
    def __init__(self, master: tk.Tk):  # Added type hint for master
        self.master = master
        master.title("Interactive Annotation Tool")
        master.geometry("1200x800")  # Adjusted size for better layout

        self.image_path: Optional[str] = None
        self.model_path: Optional[str] = None
        self.output_path: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.display_image_tk: Optional[ImageTk.PhotoImage] = None  # For Tkinter display, keep reference
        self.model: Optional[YOLO] = None
        self.results_obj: Optional[Results] = None  # Store the single Results object

        # --- GUI Layout ---
        # Control Frame
        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Image Display Frame
        self.image_frame = ttk.Frame(master, padding="10")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Widgets ---
        # Buttons
        self.btn_load_image = ttk.Button(self.control_frame, text="Load Image", command=self._load_image)
        self.btn_load_image.pack(pady=5, fill=tk.X)

        # Model Selection
        ttk.Label(self.control_frame, text="Model Path/Name:").pack(pady=(10, 0), anchor=tk.W)
        self.model_entry = ttk.Entry(self.control_frame, width=35)  # Increased width
        self.model_entry.insert(0, self.model_path or "yolov8n-seg.pt")
        self.model_entry.pack(pady=2, fill=tk.X)
        self.btn_load_model = ttk.Button(self.control_frame, text="Load Model", command=self._load_model)
        self.btn_load_model.pack(pady=5, fill=tk.X)

        # Confidence Slider
        ttk.Label(self.control_frame, text="Confidence Threshold:").pack(pady=(10, 0), anchor=tk.W)
        self.conf_slider_var = tk.DoubleVar(value=0.25)
        self.conf_slider = ttk.Scale(self.control_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.conf_slider_var, command=self._update_conf_label)
        self.conf_slider.pack(fill=tk.X, pady=2)
        self.conf_label = ttk.Label(self.control_frame, text=f"{self.conf_slider_var.get():.2f}")
        self.conf_label.pack(anchor=tk.W)

        # IoU Slider
        ttk.Label(self.control_frame, text="IoU Threshold:").pack(pady=(10, 0), anchor=tk.W)
        self.iou_slider_var = tk.DoubleVar(value=0.7)
        self.iou_slider = ttk.Scale(self.control_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.iou_slider_var, command=self._update_iou_label)
        self.iou_slider.pack(fill=tk.X, pady=2)
        self.iou_label = ttk.Label(self.control_frame, text=f"{self.iou_slider_var.get():.2f}")
        self.iou_label.pack(anchor=tk.W)

        # Run Inference Button
        self.btn_run_inference = ttk.Button(self.control_frame, text="Run Inference", command=self._run_inference,
                                            state=tk.DISABLED)
        self.btn_run_inference.pack(pady=10, fill=tk.X)

        # Save Button
        self.btn_save = ttk.Button(self.control_frame, text="Save Annotations", command=self._save_annotations,
                                   state=tk.DISABLED)
        self.btn_save.pack(pady=5, fill=tk.X)

        # Status Label (at the bottom of control frame for more space)
        self.status_label = ttk.Label(self.control_frame, text="Status: Load Image and Model", wraplength=230,
                                      justify=tk.LEFT)  # Increased wraplength
        self.status_label.pack(pady=10, side=tk.BOTTOM, fill=tk.X, expand=True)

        # Image Display Label
        self.image_label = ttk.Label(self.image_frame, text="Image will be displayed here", anchor=tk.CENTER,
                                     relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        # Ensure image_frame itself can expand
        self.image_frame.pack_propagate(False)

    def _update_conf_label(self, value_str: str):  # Value from Scale is string
        self.conf_label.config(text=f"{float(value_str):.2f}")

    def _update_iou_label(self, value_str: str):  # Value from Scale is string
        self.iou_label.config(text=f"{float(value_str):.2f}")

    def _update_button_states(self):
        if self.original_image is not None and self.model is not None:
            self.btn_run_inference.config(state=tk.NORMAL)
        else:
            self.btn_run_inference.config(state=tk.DISABLED)

        if self.results_obj is not None:  # Check the actual results object
            self.btn_save.config(state=tk.NORMAL)
        else:
            self.btn_save.config(state=tk.DISABLED)

    def _display_image_on_label(self, image_np: Optional[np.ndarray]):
        if image_np is None:
            self.image_label.config(image='', text="No image to display")
            self.display_image_tk = None  # Clear reference
            return

        # Ensure frame has a size before calculating scale
        self.master.update_idletasks()
        max_height = self.image_label.winfo_height() - 10  # Small padding
        max_width = self.image_label.winfo_width() - 10

        if max_height <= 20 or max_width <= 20:  # Frame not properly sized yet or too small
            # Fallback if winfo_height/width is not reliable initially
            # This might happen if called before window is fully drawn
            # Using a fixed reasonable size or aspect ratio from image
            img_h, img_w = image_np.shape[:2]
            aspect_ratio = img_w / img_h
            max_height = 600  # Arbitrary reasonable default
            max_width = int(max_height * aspect_ratio)

        h, w = image_np.shape[:2]
        if h == 0 or w == 0:
            self.image_label.config(image='', text="Invalid image dimensions")
            return

        scale = min(max_width / w, max_height / h) if max_width > 0 and max_height > 0 else 1.0

        display_img_resized = image_np
        if scale < 1.0:  # Only downscale, don't upscale small images
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                display_img_resized = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            image_rgb = cv2.cvtColor(display_img_resized, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(image_rgb)
            self.display_image_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.display_image_tk, text="")
        except Exception as e:
            print(f"Error converting image for display: {e}")
            self.image_label.config(image='', text="Error displaying image")
            self.display_image_tk = None

    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All files", "*.*"))
        )
        if not path:
            return

        self.image_path = path
        self.original_image = load_image(self.image_path)
        self.results_obj = None  # Reset results when new image is loaded
        self.annotated_image = None

        if self.original_image is not None:
            self.status_label.config(text=f"Status: Image loaded\n{os.path.basename(self.image_path)}")
            # Display original image immediately
            self.master.after(50, lambda: self._display_image_on_label(self.original_image))
        else:
            self.status_label.config(text="Status: Failed to load image.")
            messagebox.showerror("Error", f"Failed to load image from {self.image_path}")
            self._display_image_on_label(None)

        self._update_button_states()

    def _load_model(self):
        path = self.model_entry.get()
        if not path:
            messagebox.showerror("Error", "Please enter a model path or name.")
            return

        self.status_label.config(text="Status: Loading model...")
        self.master.update_idletasks()

        self.model_path = path
        self.model = load_model(self.model_path, device='auto')

        if self.model is not None:
            self.status_label.config(
                text=f"Status: Model loaded\n{os.path.basename(self.model_path or 'Unknown Model')}")
        else:
            self.status_label.config(text="Status: Failed to load model.")
            messagebox.showerror("Error", f"Failed to load model: {self.model_path}")

        self._update_button_states()

    def _run_inference(self):
        if self.original_image is None or self.model is None:
            messagebox.showerror("Error", "Please load an image and a model first.")
            return

        conf = self.conf_slider_var.get()
        iou = self.iou_slider_var.get()

        self.status_label.config(text="Status: Running inference...")
        self.master.update_idletasks()

        # Run inference (now expects a single Results object or None)
        self.results_obj = run_inference(self.model, self.original_image, conf, iou)

        if self.results_obj:
            self.annotated_image = visualize_results(self.original_image, self.results_obj)
            num_dets = len(self.results_obj.boxes) if self.results_obj.boxes is not None else 0
            self.status_label.config(text=f"Status: Inference complete. Found {num_dets} objects.")
            self._display_image_on_label(self.annotated_image)
        else:
            self.annotated_image = None  # Clear previous annotated image
            self.status_label.config(text="Status: Inference failed or no objects found.")
            messagebox.showwarning("Inference",
                                   "Inference process completed, but no objects were detected or an error occurred.")
            self._display_image_on_label(self.original_image)  # Show original image if inference yields nothing

        self._update_button_states()

    def _save_annotations(self):
        if self.results_obj is None or self.original_image is None:
            messagebox.showerror("Error", "No annotations to save. Run inference first.")
            return

        base = os.path.basename(self.image_path or "untitled.png")
        name, _ = os.path.splitext(base)
        default_name = f"{name}_annotations.json"

        path = filedialog.asksaveasfilename(
            title="Save Annotations As",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=(("JSON files", "*.json"), ("All files", "*.*"))
        )

        if not path:
            return

        self.output_path = path
        success = save_annotations(self.results_obj, self.output_path, self.original_image.shape)

        if success:
            self.status_label.config(text=f"Status: Annotations saved to\n{os.path.basename(self.output_path)}")
            messagebox.showinfo("Success", f"Annotations saved to {self.output_path}")
        else:
            self.status_label.config(text="Status: Failed to save annotations.")
            messagebox.showerror("Error", "Failed to save annotations.")


# --- Alternative Manual Visualization (using OpenCV - commented out) ---
# This provides more control but requires more code.
#
# def visualize_results_manual(image: np.ndarray, results_obj: Optional[Results]) -> np.ndarray:
#     if image is None: return np.zeros((100,100,3), dtype=np.uint8)
#     if results_obj is None: return image.copy()
#
#     vis_image = image.copy()
#     boxes_data = results_obj.boxes
#     masks_data = results_obj.masks
#     names = results_obj.names
#
#     if boxes_data is None: return vis_image
#
#     pixel_boxes = boxes_data.xyxy.cpu().numpy() # Bounding boxes (x1, y1, x2, y2)
#     confs = boxes_data.conf.cpu().numpy() # Confidences
#     class_ids = boxes_data.cls.cpu().numpy().astype(int) # Class IDs
#
#     for i in range(len(pixel_boxes)):
#         x1, y1, x2, y2 = map(int, pixel_boxes[i])
#         conf = confs[i]
#         cls_id = class_ids[i]
#         label = f"{names.get(cls_id, f'cls_{cls_id}')} {conf:.2f}" if names else f"cls_{cls_id} {conf:.2f}"
#
#         # Draw bounding box
#         color = (0, 255, 0) # Green
#         cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
#
#         # Draw label
#         (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(vis_image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), color, cv2.FILLED)
#         cv2.putText(vis_image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
#
#         # Draw mask (if available and is segmentation model)
#         if masks_data is not None and masks_data.data is not None and len(masks_data.data) > i:
#             mask_raw = masks_data.data[i].cpu().numpy().astype(np.uint8)
#             # Ultralytics masks are usually the size of the network input or original image, check 'masks.orig_shape'
#             # Resize mask to image size if needed
#             if mask_raw.shape != vis_image.shape[:2]:
#                  mask_resized = cv2.resize(mask_raw, (vis_image.shape[1], vis_image.shape[0]), interpolation=cv2.INTER_NEAREST) # Corrected resize
#             else:
#                  mask_resized = mask_raw
#
#             colored_mask_overlay = np.zeros_like(vis_image, dtype=np.uint8)
#             # Create a color for the mask, e.g., based on class_id or a fixed color
#             mask_color = (255, 0, 255) # Magenta example
#             colored_mask_overlay[mask_resized > 0] = mask_color
#             # Blend mask with image
#             vis_image = cv2.addWeighted(vis_image, 1.0, colored_mask_overlay, 0.5, 0)
#
#     print("Manual visualization generated.")
#     return vis_image

# --- Main Execution ---
def create_annotation_gui():
    """Creates and runs the main Tkinter GUI."""
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    print("Starting Interactive Annotation Tool...")
    # Optional: Add argparse here to pass initial image/model paths via command line
    create_annotation_gui()
    print("Application closed.")