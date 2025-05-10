import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import json
import torch
import re  # For parsing edited coordinates

from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List, Optional


# --- Helper Functions ---

def load_image(image_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return None
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Failed to load image (invalid format or file): {image_path}")
            return None
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def load_model(model_path: str, device: str = 'auto') -> Optional[YOLO]:
    resolved_device = device
    if device == 'auto':
        if torch.cuda.is_available():
            resolved_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
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
        model.predict_device = resolved_device
        print(f"Model '{model_path}' loaded successfully. Will attempt inference on '{resolved_device}'.")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def run_inference(model: YOLO, image: np.ndarray,
                  confidence_threshold: float = 0.25,
                  iou_threshold: float = 0.7) -> Optional[Results]:
    if model is None or image is None:
        print("Error: Model or image is not loaded for inference.")
        return None
    try:
        device_to_use = getattr(model, 'predict_device', 'cpu')
        results_list = model.predict(
            source=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            device=device_to_use,
            verbose=False
        )
        if results_list and len(results_list) > 0:
            return results_list[0]
        return None
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def visualize_results(image: np.ndarray, results_obj: Optional[Results],
                      edited_pixel_boxes: Optional[List[List[int]]] = None) -> np.ndarray:
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder

    img_to_annotate = image.copy()

    if results_obj is None and not edited_pixel_boxes:
        return img_to_annotate  # Return copy if nothing to draw

    # If edited_pixel_boxes are provided, we'll try to draw them.
    # This version does NOT use edited_pixel_boxes for visualization, sticks to results_obj.plot()
    # For visual feedback of edited boxes, this function would need to be significantly changed
    # to draw boxes manually using cv2.rectangle based on edited_pixel_boxes.
    # For now, visualization is based on the original detection results.
    if results_obj:
        try:
            annotated_image = results_obj.plot(img=img_to_annotate)  # Plot on a copy
            return annotated_image
        except Exception as e:
            print(f"Error during results_obj.plot() visualization: {e}")
            # Fallback to drawing manually if plot fails or if we want to prioritize edited boxes
            pass  # Continue to manual drawing if needed, or just return img_to_annotate

    # Fallback or if we wanted to draw edited_pixel_boxes (not fully implemented here for visualization)
    # This is where you would draw the `edited_pixel_boxes` if you wanted them reflected in the preview.
    # For now, this function primarily relies on results_obj.plot().
    return img_to_annotate


def save_annotations(
        results_obj: Optional[Results],
        output_path: str,
        image_shape: tuple,
        image_filename: str,
        distance: Optional[str] = None,
        edited_pixel_boxes: Optional[List[List[int]]] = None,
        image_level_object_type: Optional[str] = None
) -> bool:
    if not results_obj and not edited_pixel_boxes:  # Need at least original results or some edited boxes
        print(f"Warning: No results or edited boxes to save for {image_filename}.")
        return False

    height, width, _ = image_shape
    annotations_list = []
    final_pixel_boxes = []

    # Determine which pixel boxes to use
    use_edited_boxes = False
    if edited_pixel_boxes is not None:
        if results_obj and results_obj.boxes and len(edited_pixel_boxes) == len(results_obj.boxes):
            final_pixel_boxes = edited_pixel_boxes
            use_edited_boxes = True
            print(f"Using {len(edited_pixel_boxes)} edited pixel boxes for saving.")
        elif not results_obj or not results_obj.boxes:  # No original boxes, use edited if any
            final_pixel_boxes = edited_pixel_boxes
            use_edited_boxes = True
            print(f"Using {len(edited_pixel_boxes)} edited pixel boxes (no original detections to compare).")
        else:  # Mismatch in number of boxes
            print(f"Warning: Number of edited boxes ({len(edited_pixel_boxes)}) "
                  f"does not match original detections ({len(results_obj.boxes) if results_obj and results_obj.boxes else 0}). "
                  "Saving original pixel boxes.")
            if results_obj and results_obj.boxes and results_obj.boxes.xyxy is not None:
                final_pixel_boxes = results_obj.boxes.xyxy.cpu().numpy().astype(int).tolist()
            else:  # No original pixel boxes to fall back on
                print(f"Error: Cannot determine pixel boxes for {image_filename}.")
                return False  # Or save with empty pixel boxes if other data is valuable
    elif results_obj and results_obj.boxes and results_obj.boxes.xyxy is not None:
        final_pixel_boxes = results_obj.boxes.xyxy.cpu().numpy().astype(int).tolist()
    # If neither edited nor original pixel boxes are available, final_pixel_boxes remains empty.

    # Primary loop based on original detections if they exist, for classes, confidences, normalized coords
    if results_obj and results_obj.boxes:
        names = results_obj.names
        norm_boxes_xyxyn = results_obj.boxes.xyxyn.cpu().numpy() if results_obj.boxes.xyxyn is not None else [
                                                                                                                 None] * len(
            results_obj.boxes)
        confs = results_obj.boxes.conf.cpu().numpy() if results_obj.boxes.conf is not None else [0.0] * len(
            results_obj.boxes)
        class_ids = results_obj.boxes.cls.cpu().numpy().astype(int) if results_obj.boxes.cls is not None else [
                                                                                                                  -1] * len(
            results_obj.boxes)
        masks_data = results_obj.masks

        num_detections = len(results_obj.boxes)

        for i in range(num_detections):
            class_id = int(class_ids[i])
            class_name = names.get(class_id, f"class_{class_id}") if names and class_id != -1 else f"class_{class_id}"

            current_pixel_box = final_pixel_boxes[i] if i < len(
                final_pixel_boxes) else None  # Get corresponding pixel box

            annotation = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confs[i]),
                "bbox_normalized_xyxyn": norm_boxes_xyxyn[i].tolist() if norm_boxes_xyxyn[i] is not None else None,
                "bbox_pixels_xyxy": current_pixel_box
            }
            if masks_data is not None and masks_data.xy is not None and len(masks_data.xy) > i:
                annotation["segmentation_normalized_polygons"] = masks_data.xy[i].tolist()
            annotations_list.append(annotation)

    elif use_edited_boxes and final_pixel_boxes:  # No original results, but we have edited boxes
        print("Saving annotations based purely on edited boxes (no class/confidence from model).")
        for i, box_coords in enumerate(final_pixel_boxes):
            annotation = {
                "class_id": -1,  # Placeholder
                "class_name": "unknown (edited)",  # Placeholder
                "confidence": 0.0,  # Placeholder
                "bbox_normalized_xyxyn": None,  # Cannot derive from pixels without image shape at this exact point
                "bbox_pixels_xyxy": box_coords
            }
            annotations_list.append(annotation)

    output_data = {
        "image_filename": image_filename,
        "image_height": height,
        "image_width": width,
        "image_specific_distance": distance if distance and distance.strip() else "Not provided",
        "image_level_object_type": image_level_object_type if image_level_object_type else "Unspecified",
        "annotations": annotations_list
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Annotations saved successfully for {image_filename} to: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"Error saving annotations for {image_filename} to {output_path}: {e}")
        return False


class AnnotationApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Interactive Annotation Tool")
        master.geometry("1300x900")  # Adjusted size

        self.image_directory: Optional[str] = None
        self.image_files: List[str] = []
        self.current_image_index: int = -1
        self.current_image_path: Optional[str] = None

        self.model_path: Optional[str] = None
        self.original_image: Optional[np.ndarray] = None
        self.annotated_image: Optional[np.ndarray] = None
        self.display_image_tk: Optional[ImageTk.PhotoImage] = None
        self.model: Optional[YOLO] = None
        self.results_obj: Optional[Results] = None

        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.image_frame = ttk.Frame(master, padding="10")
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Directory and Image Navigation
        self.btn_load_dir = ttk.Button(self.control_frame, text="Load Directory", command=self._load_directory)
        self.btn_load_dir.pack(pady=5, fill=tk.X)

        self.navigation_frame = ttk.Frame(self.control_frame)
        self.navigation_frame.pack(pady=2, fill=tk.X)
        self.btn_prev_image = ttk.Button(self.navigation_frame, text="<< Previous", command=self._previous_image,
                                         state=tk.DISABLED)
        self.btn_prev_image.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 1))
        self.btn_next_image = ttk.Button(self.navigation_frame, text="Next >>", command=self._next_image,
                                         state=tk.DISABLED)
        self.btn_next_image.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(1, 0))

        self.current_file_label = ttk.Label(self.control_frame, text="No directory loaded.", wraplength=240,
                                            justify=tk.LEFT)
        self.current_file_label.pack(pady=(2, 5), fill=tk.X)

        # Model Selection
        ttk.Label(self.control_frame, text="Model Path/Name:").pack(pady=(5, 0), anchor=tk.W)
        self.model_entry = ttk.Entry(self.control_frame, width=38)
        self.model_entry.insert(0, self.model_path or "yolov8n.pt")  # Changed default to non-seg
        self.model_entry.pack(pady=2, fill=tk.X)
        self.btn_load_model = ttk.Button(self.control_frame, text="Load Model", command=self._load_model)
        self.btn_load_model.pack(pady=(2, 5), fill=tk.X)

        # Parameters
        ttk.Label(self.control_frame, text="Confidence Threshold:").pack(pady=(5, 0), anchor=tk.W)
        self.conf_slider_var = tk.DoubleVar(value=0.25)
        self.conf_slider = ttk.Scale(self.control_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                                     variable=self.conf_slider_var, command=self._update_conf_label)
        self.conf_slider.pack(fill=tk.X, pady=1)
        self.conf_label = ttk.Label(self.control_frame, text=f"{self.conf_slider_var.get():.2f}")
        self.conf_label.pack(anchor=tk.W, pady=(0, 5))

        ttk.Label(self.control_frame, text="IoU Threshold:").pack(pady=(5, 0), anchor=tk.W)
        self.iou_slider_var = tk.DoubleVar(value=0.45)  # Default IoU
        self.iou_slider = ttk.Scale(self.control_frame, from_=0.01, to=1.0, orient=tk.HORIZONTAL,
                                    variable=self.iou_slider_var, command=self._update_iou_label)
        self.iou_slider.pack(fill=tk.X, pady=1)
        self.iou_label = ttk.Label(self.control_frame, text=f"{self.iou_slider_var.get():.2f}")
        self.iou_label.pack(anchor=tk.W, pady=(0, 5))

        # Actions
        self.btn_run_inference = ttk.Button(self.control_frame, text="Run Inference", command=self._run_inference,
                                            state=tk.DISABLED)
        self.btn_run_inference.pack(pady=5, fill=tk.X)

        # Object Type Dropdown
        ttk.Label(self.control_frame, text="Image Object Type:").pack(pady=(5, 0), anchor=tk.W)
        self.object_type_var = tk.StringVar(value="Unspecified")
        self.object_type_options = ["Unspecified", "Person", "Car", "Animal", "Other"]
        self.object_type_combobox = ttk.Combobox(self.control_frame, textvariable=self.object_type_var,
                                                 values=self.object_type_options, state="readonly", width=35)
        self.object_type_combobox.pack(pady=2, fill=tk.X)
        self.object_type_combobox.set("Unspecified")  # Default value

        # Distance Input
        ttk.Label(self.control_frame, text="Distance for Image (e.g., meters):").pack(pady=(5, 0), anchor=tk.W)
        self.distance_var = tk.StringVar()
        self.distance_entry = ttk.Entry(self.control_frame, textvariable=self.distance_var, width=38)
        self.distance_entry.pack(pady=2, fill=tk.X)

        # Coordinates Display (Editable)
        ttk.Label(self.control_frame, text="Bounding Box Coords (x1,y1,x2,y2) - Editable:").pack(pady=(5, 0),
                                                                                                 anchor=tk.W)
        self.coord_display_text = scrolledtext.ScrolledText(self.control_frame, height=7, width=38, wrap=tk.WORD,
                                                            relief=tk.SUNKEN)
        self.coord_display_text.pack(pady=2, fill=tk.X, expand=False)
        self.coord_display_text.configure(state=tk.DISABLED)

        self.btn_save = ttk.Button(self.control_frame, text="Save Annotations", command=self._save_annotations,
                                   state=tk.DISABLED)
        self.btn_save.pack(pady=5, fill=tk.X)

        # Status Label
        self.status_label = ttk.Label(self.control_frame, text="Status: Load Directory and Model", wraplength=240,
                                      justify=tk.LEFT)
        self.status_label.pack(pady=(10, 0), side=tk.BOTTOM, fill=tk.X)

        # Image Display Label
        self.image_label = ttk.Label(self.image_frame, text="Load a directory to display images", anchor=tk.CENTER,
                                     relief=tk.SUNKEN)
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_frame.pack_propagate(False)
        self._update_button_states()

    def _update_conf_label(self, value_str: str):
        self.conf_label.config(text=f"{float(value_str):.2f}")

    def _update_iou_label(self, value_str: str):
        self.iou_label.config(text=f"{float(value_str):.2f}")

    def _clear_image_specific_data(self):
        self.results_obj = None
        self.annotated_image = None
        self.distance_var.set("")
        self.object_type_var.set("Unspecified")  # Reset dropdown
        self.coord_display_text.configure(state=tk.NORMAL)
        self.coord_display_text.delete('1.0', tk.END)
        self.coord_display_text.configure(state=tk.DISABLED)

    def _update_button_states(self):
        can_infer = self.original_image is not None and self.model is not None
        self.btn_run_inference.config(state=tk.NORMAL if can_infer else tk.DISABLED)
        can_save = self.results_obj is not None and self.current_image_path is not None
        self.btn_save.config(
            state=tk.NORMAL if can_save else tk.DISABLED)  # Also enable if there are manually entered coords? For now, depends on results_obj.
        has_images = self.image_files and len(self.image_files) > 0
        self.btn_prev_image.config(state=tk.NORMAL if has_images else tk.DISABLED)
        self.btn_next_image.config(state=tk.NORMAL if has_images else tk.DISABLED)

    def _display_image_on_label(self, image_np: Optional[np.ndarray]):
        if image_np is None:
            self.image_label.config(image='', text="No image to display")
            self.display_image_tk = None
            return
        self.master.update_idletasks()
        max_h, max_w = self.image_label.winfo_height() - 10, self.image_label.winfo_width() - 10
        if max_h <= 20 or max_w <= 20: max_h, max_w = 800, 1000  # Fallback
        h, w = image_np.shape[:2]
        if h == 0 or w == 0: return
        scale = 1.0
        if w > max_w or h > max_h: scale = min(max_w / w, max_h / h)
        disp_img = image_np
        if 0 < scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            if nw > 0 and nh > 0: disp_img = cv2.resize(image_np, (nw, nh), cv2.INTER_AREA)
        try:
            img_rgb = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.display_image_tk = ImageTk.PhotoImage(img_pil)
            self.image_label.config(image=self.display_image_tk, text="")
        except Exception as e:
            print(f"Error converting image for display: {e}")
            self.image_label.config(image='', text="Error displaying image")
            self.display_image_tk = None

    def _load_directory(self):
        path = filedialog.askdirectory(title="Select Image Directory")
        if not path: return
        self.image_directory = path
        self.image_files = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
        if not self.image_files:
            self.current_image_index, self.current_image_path, self.original_image = -1, None, None
            self._clear_image_specific_data()
            self._display_image_on_label(None)
            self.current_file_label.config(text=f"No images found in:\n{os.path.basename(path)}")
            messagebox.showinfo("No Images", "No compatible image files found.")
        else:
            self.current_image_index = 0
            self._load_current_image_from_list()
        self._update_button_states()

    def _load_current_image_from_list(self):
        if not self.image_directory or not self.image_files or \
                not (0 <= self.current_image_index < len(self.image_files)):
            self.current_file_label.config(text="Invalid image selection.")
            self.original_image, self.current_image_path = None, None
            self._display_image_on_label(None)
            self._clear_image_specific_data()
            self._update_button_states()
            return

        filename = self.image_files[self.current_image_index]
        self.current_image_path = os.path.join(self.image_directory, filename)
        self.original_image = load_image(self.current_image_path)
        self._clear_image_specific_data()
        if self.original_image is not None:
            self.status_label.config(text="Status: Image loaded.")
            self.current_file_label.config(text=f"{filename}\n({self.current_image_index + 1}/{len(self.image_files)})")
            self.master.after(50, lambda: self._display_image_on_label(self.original_image))
        else:
            self.status_label.config(text=f"Status: Failed to load {filename}.")
            self.current_file_label.config(text=f"Failed: {filename}")
            self._display_image_on_label(None)
        self._update_button_states()

    def _next_image(self):
        if not self.image_files: return
        self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
        self._load_current_image_from_list()

    def _previous_image(self):
        if not self.image_files: return
        self.current_image_index = (self.current_image_index - 1 + len(self.image_files)) % len(self.image_files)
        self._load_current_image_from_list()

    def _load_model(self):
        path = self.model_entry.get()
        if not path: messagebox.showerror("Error", "Model path/name empty."); return
        self.status_label.config(text="Status: Loading model...")
        self.master.update_idletasks()
        self.model_path = path
        self.model = load_model(self.model_path, device='auto')
        self.status_label.config(
            text=f"Status: Model {'loaded' if self.model else 'failed to load'}: {os.path.basename(self.model_path or '')}")
        if not self.model: messagebox.showerror("Error", f"Failed to load model: {self.model_path}")
        self._update_button_states()

    def _parse_edited_coordinates(self, text_content: str) -> Optional[List[List[int]]]:
        parsed_boxes = []
        lines = text_content.strip().split('\n')
        coord_pattern = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue  # Skip empty lines
            match = coord_pattern.search(line)
            if match:
                try:
                    coords = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                    parsed_boxes.append(coords)
                except ValueError:
                    messagebox.showwarning("Coordinate Parse Error",
                                           f"Non-integer coordinate in line {i + 1}: '{line}'. Please use integers.")
                    return None
            elif not line.lower().startswith("no boxes detected"):  # Allow "No boxes detected."
                messagebox.showwarning("Coordinate Format Error",
                                       f"Invalid format in line {i + 1}: '{line}'. Expected 'Box N: [x1,y1,x2,y2]' or just '[x1,y1,x2,y2]'.")
                return None
        return parsed_boxes

    def _display_coordinates(self, results_obj: Optional[Results]):
        self.coord_display_text.configure(state=tk.NORMAL)
        self.coord_display_text.delete('1.0', tk.END)
        if results_obj and results_obj.boxes:
            boxes_xyxy = results_obj.boxes.xyxy.cpu().numpy().astype(int)
            if len(boxes_xyxy) == 0:
                self.coord_display_text.insert(tk.END, "No boxes detected.")
            else:
                for i, box in enumerate(boxes_xyxy):
                    self.coord_display_text.insert(tk.END, f"Box {i + 1}: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]\n")
        else:
            self.coord_display_text.insert(tk.END,
                                           "No results or boxes to display. You can manually enter coordinates here in the format [x1,y1,x2,y2] per line.")
        # Keep NORMAL if results are present, to allow editing.
        # self.coord_display_text.configure(state=tk.DISABLED) # Only disable if no results? Or always allow editing?

    def _run_inference(self):
        if self.original_image is None or self.model is None:
            messagebox.showerror("Error", "Load image (from directory) and model first.")
            return
        conf, iou = self.conf_slider_var.get(), self.iou_slider_var.get()
        self.status_label.config(text="Status: Running inference...");
        self.master.update_idletasks()
        self.results_obj = run_inference(self.model, self.original_image, conf, iou)
        self._display_coordinates(self.results_obj)  # Populate text box, implicitly enables for editing.
        if self.results_obj:
            self.annotated_image = visualize_results(self.original_image,
                                                     self.results_obj)  # Visualization uses original results
            num_dets = len(self.results_obj.boxes) if self.results_obj.boxes is not None else 0
            self.status_label.config(text=f"Status: Inference complete. Found {num_dets} objects.")
            self._display_image_on_label(self.annotated_image)
        else:
            self.annotated_image = None
            self.status_label.config(text="Status: Inference failed or no objects found.")
            self._display_image_on_label(self.original_image)
        self._update_button_states()

    def _save_annotations(self):
        if self.original_image is None or self.current_image_path is None:
            messagebox.showerror("Error", "No image context to save. Load an image from a directory.")
            return
        # `results_obj` can be None if user is manually entering coordinates after failed inference
        # However, our save function and general logic relies heavily on results_obj for now.
        # Let's enforce that results_obj must exist OR there must be valid text in coord_display_text

        edited_coords_text = self.coord_display_text.get("1.0", tk.END).strip()
        parsed_edited_boxes = self._parse_edited_coordinates(edited_coords_text)

        if parsed_edited_boxes is None:  # Parsing error occurred and was signaled by None
            messagebox.showerror("Save Error",
                                 "Could not save due to coordinate parsing errors. Please check the format in the coordinates text box.")
            return

        # If no original results, but user entered boxes, we might allow saving.
        # However, the current save_annotations function ties classes/confidences to original results.
        # For this iteration, let's keep the dependency on having original results for a meaningful save,
        # unless we explicitly design for "manual annotation from scratch".
        if self.results_obj is None and not parsed_edited_boxes:
            messagebox.showerror("Error", "No inference results to save and no manual coordinates entered.")
            return
        if self.results_obj is None and parsed_edited_boxes:
            print("Attempting to save with manually entered coordinates only.")
            # The save_annotations function has some logic for this.

        output_dir = self.image_directory if self.image_directory else os.path.expanduser("~")
        base_img_fname = os.path.basename(self.current_image_path)
        name_part, _ = os.path.splitext(base_img_fname)
        default_json_fname = f"{name_part}_annotations.json"
        initial_save_path = os.path.join(output_dir, default_json_fname)

        output_json_path = filedialog.asksaveasfilename(
            title="Save Annotations As", initialdir=output_dir, initialfile=default_json_fname,
            defaultextension=".json", filetypes=(("JSON files", "*.json"), ("All files", "*.*")))
        if not output_json_path: return

        user_distance = self.distance_var.get()
        user_object_type = self.object_type_var.get()

        success = save_annotations(
            self.results_obj, output_json_path, self.original_image.shape,
            base_img_fname, user_distance, parsed_edited_boxes, user_object_type)

        if success:
            self.status_label.config(text=f"Status: Annotations saved to\n{os.path.basename(output_json_path)}")
        else:
            self.status_label.config(text="Status: Failed to save annotations.")
            # More specific error already printed by save_annotations or shown by messagebox


def create_annotation_gui():
    root = tk.Tk()
    app = AnnotationApp(root)
    root.mainloop()


if __name__ == "__main__":
    print("Starting Interactive Annotation Tool...")
    create_annotation_gui()
    print("Application closed.")