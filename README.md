
# 📸 Investigation of Alternative Distance and Object Detection Using a Single Camera Warped Lens 🔭

## 🎯 Project Overview

This project, Taken for CS559; Computer Vision, explores an alternative method for object distance detection using a single, modified camera lens and an AI model. The core idea was to simulate a warped lens effect by applying a synthetic blur to images and then train a YOLO V3 model to perform object detection and distance estimation on this distorted data. The goal was to investigate if such a system could offer a more cost-effective alternative to traditional methods like LiDAR or stereoscopic cameras.

The research involved:
1.  Developing an algorithm to synthetically apply a depth-aware blur to an existing image dataset (simulating a camera with an inclined sensor).
2.  Modifying and fine-tuning a YOLO V3-based object detection model (Dist-YOLO) to predict object classes (cars, cyclists, pedestrians) and their distances.
3.  Training two versions of the model: one on the original, unaltered dataset and another on the synthetically blurred dataset.
4.  Comparing the performance of these models to assess the viability of the proposed approach.

Ultimately, the project found that while the concept is intriguing, the YOLO V3 model's performance significantly degraded when trained and/or tested on the synthetically blurred images. The image distortion, even when the model was fine-tuned on it, led to a substantial loss in detection accuracy and confidence.

## 🔑 Key Objectives

* To simulate the optical characteristics of a modified camera (warped lens/inclined sensor) through synthetic image blurring.
* To adapt an object detection model (YOLO V3) to estimate object distances from single monocular images.
* To evaluate the feasibility of using intentionally distorted images for distance estimation as a low-cost alternative to existing technologies.
* To compare the performance of models trained on blurred vs. non-blurred image data.

## 🛠️ Methodology

### 1. 🖼️ Synthetic Data Generation (Simulating a Modified Camera)
Due to challenges in obtaining consistent and usable data from a physically modified camera, a synthetic approach was adopted. An algorithm was developed to apply a gradient blur to images from a standard dataset (e.g., KITTI). This blur was designed to simulate the effect of a camera with an inclined image sensor, where different parts of the image would have varying degrees of focus based on distance.

* **Relevant Code:** The implementation of this blurring algorithm can be found in the Jupyter Notebook:
    * `Bluring Alogirthm For Simulating Modified Camera.ipynb`

### 2. 🧠 Model Architecture and Training
The project utilized a YOLO V3-based architecture, building upon concepts from Dist-YOLO, which integrates distance estimation with object detection. The inital editing and weights of the model was provided by the yolo-dist repository cited bellow. The model was modified to:
* Predict object classes (cars, cyclists, pedestrians).
* Estimate the distance to each detected object.
* Handle the additional distance dimension in the training labels.

Two primary models were trained:
* **Model 1 (Baseline):** Fine-tuned on the original, unaltered image dataset.
* **Model 2 (Blurred):** Fine-tuned on the synthetically blurred image dataset.

Training involved careful consideration of loss functions, particularly re-weighting the confidence loss to manage issues arising from the blurred data and reduced class set.

* **Relevant Code:** The model definition, training procedures for both models (baseline and blurred), and fine-tuning specifics are detailed in:
    * `train both models.ipynb`

### 3. 📊 Performance Evaluation
The trained models were evaluated on their ability to detect objects and estimate their distances accurately. Metrics such as F1-score, precision, recall (for object detection), and Mean Absolute Error (MAE) (for distance estimation) were used for comparison.

* **Relevant Code:** The scripts and analyses for comparing the performance of the two models, including generation of comparative plots and metrics, can be found in:
    * `compare.ipynb`

## ⚙️ Setup and Usage

This section outlines the steps to set up the necessary environment and run the Jupyter Notebooks associated with this project.

1.  **Clone the Repository and Submodule:**
    ```bash
    git clone [https://github.com/TeagueSS/ComputerVisionFinal.git](https://github.com/TeagueSS/ComputerVisionFinal.git)
    cd ComputerVisionFinal
    mkdir ExistingModel 
    cd ExistingModel
    git clone [https://gitlab.com/EnginCZ/yolo-with-distance.git](https://gitlab.com/EnginCZ/yolo-with-distance.git)
    cd .. 
    ```
2.  **Set up a Python Environment:**
    It is highly recommended to use a virtual environment. The author (Teague) used Anaconda.
    ```bash
    # Using Anaconda
    conda create -n cv_final_env python=3.8  # Or your preferred Python version
    conda activate cv_final_env
    ```
    Alternatively, using `venv`:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    All required Python packages are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `tensorflow-macos==2.13.0` (or `tensorflow` for non-Mac M-series), `keras==2.13.1`, `opencv-python==4.11.0.86`, `numpy==1.24.3`, `matplotlib==3.7.5`, `jupyter==1.0.0`, `pandas==2.0.3`, `ultralytics==8.3.131`.

    *Note for `yolo-with-distance` submodule:* The `yolo-with-distance` repository might have its own specific dependencies or setup instructions. Please refer to its documentation within the `ExistingModel/yolo-with-distance` directory.

4.  **Download Datasets (if applicable):**
    This project primarily uses the KITTI dataset.
    * **KITTI Dataset:** You will need to download the relevant parts of the KITTI Vision Benchmark Suite, specifically the data for object detection (images and labels).
        * Homepage: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
        * Object detection data: [http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)
    * Place the downloaded and extracted dataset in a directory structure that your notebooks expect (e.g., a `dataset/kitti/` folder in the project root or as configured in the notebooks). Please verify the exact paths used in the Jupyter notebooks.
    *(Please add more specific details here if the notebooks expect a very particular structure or if specific preprocessing steps for the dataset are required before running the notebooks.)*

5.  **Run Jupyter Notebooks:**
    Launch Jupyter Lab or Jupyter Notebook from your activated environment:
    ```bash
    jupyter lab
    # or
    # jupyter notebook
    ```
    Then, navigate to and open the following notebooks in the suggested order or as needed:
    * `Bluring Alogirthm For Simulating Modified Camera.ipynb`: To understand and run the synthetic blur generation.
    * `train both models.ipynb`: To train the baseline and blurred YOLO models. (Note: Training may require significant computational resources and time, as well as correctly configured paths to your dataset and the `yolo-with-distance` submodule).
    * `compare.ipynb`: To evaluate and compare the performance of the trained models.

6.  **Pre-trained Models (if available):**
    The original `yolo-with-distance` repository may provide pre-trained weights for YOLOv3. Refer to its documentation for download links and instructions on how to use them.
    If this project provides its own fine-tuned model weights (e.g., `yolov3_custom_final.weights` or similar `.h5` files for Keras), they should be placed in a designated models directory (e.g., `models/` or `ExistingModel/yolo-with-distance/weights/`) as expected by the notebooks.
    *(Please specify where any custom pre-trained model weights generated by this project should be placed or how they can be obtained.)*

*(Please review and fill in any remaining placeholder details above, especially concerning dataset paths and pre-trained model locations specific to your project's implementation.)*

## 📜 Key Findings & Conclusion

> * **Impact of Blur:** The synthetic blur significantly hampered the YOLO V3 model's performance. The model trained on blurred data failed to achieve reliable object detection or accurate distance estimation, even when tested on blurred images it was trained on.
> * **Baseline Model Performance:** The model trained on normal (unblurred) data performed reasonably well on unblurred test images but showed a marked decrease in accuracy when presented with blurred images.
> * **Limitations of YOLO V3:** The study suggests that older models like YOLO V3 may not be robust enough to handle significant image distortions, even with fine-tuning. The loss of image quality due to blurring led to a critical loss of confidence in the model's predictions.
> * **Overall:** The project concluded that the tested approach of using a single warped lens (simulated by blur) with a YOLO V3 model was not a successful method for reliable distance estimation.

## ⚠️ Challenges
* Initial attempts to collect real-world data with a physically modified camera proved difficult due to hardware limitations and severe image quality issues (excessive blur, color distortion).
* Training models on blurred data led to issues like exploding loss values and poor convergence, requiring careful adjustments to the training process and loss functions.

## 🚀 Future Work
* Investigate the use of more modern and robust object detection architectures (e.g., newer versions of YOLO like YOLOv8 or YOLOv11, or Transformer-based models) which might be more resilient to image distortions.
* Explore more sophisticated data augmentation techniques or domain adaptation methods to better train models for distorted image conditions.
* Further research into the fundamental problem of extracting reliable depth cues from intentionally degraded monocular images.

## 📚 References (from the research paper)
1.  Liu, X., Tian, Q., Wanchun, C., & Xingliang, Y. (2010). Real-time distance measurement using a modified camera. *2010 IEEE Sensors Applications Symposium (SAS)*.
2.  Vajgl, M., Hurtik, P., & Nejezchleba, T. (2022). Dist-YOLO: Fast Object Detection with Distance Estimation. *Applied Sciences, 12*(3), 1354.
3.  Marek Vajgl / Yolo With Distance GitLab. (2024). https://gitlab.com/EnginCZ/yolo-with-distance
4.  The KITTI Vision Benchmark Suite. www.cvlibs.net. https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

## 🧑‍💻 Contributors (as per the paper)
* Teague Sangster
* Simmon Chura
* Elijah Pearce
* Ryan Renales
* Jwann Sy


---
