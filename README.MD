# Face Alignment Comparison

This Python script compares three different methods of face alignment—using **face-alignment library**, **mediapipe**, and **dlib**. It takes images from a specified input folder, performs face alignment using each method, and generates a combined result image for visual comparison.

## Prerequisites

Make sure you have the following libraries installed:

- OpenCV (`cv2`)
- dlib
- face_alignment
- mediapipe
- numpy
- os

```bash
pip install opencv-python dlib face_alignment mediapipe numpy
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/hamidreza07/face-alignment-comparison.git
   cd face-alignment-comparison
   ```

2. Add your input images to the `images` folder.

3. Run the script:

   ```bash
   python face_alignment_comparison.py
   ```

4. View the results in the `results_combined` folder.

## Methodology

The script uses three different face alignment methods:

1. **face-alignment library (fa):** Utilizes the face-alignment library to detect and align 68 facial landmarks.

2. **mediapipe (mp):** Uses the mediapipe library to detect and align facial landmarks.

3. **dlib:** Employs the dlib library for frontal face detection and 68-point landmark alignment.

The results are combined into a single image for each input, showcasing the original image along with the aligned faces using each method.

## Folder Structure

- **images:** Input folder containing images for face alignment.
- **results_combined:** Output folder containing the combined result images.

## Notes

- Ensure that the input images have valid formats (JPEG, PNG) for processing.

- The script resizes images for consistent display and places the aligned faces and titles at the bottom of the canvas.

- Titles are added for each alignment method, and a title for the original image is placed at the top.

## Acknowledgments

- [OpenCV](https://github.com/opencv/opencv)
- [dlib](https://github.com/davisking/dlib)
- [face-alignment](https://github.com/1adrianb/face-alignment)
- [mediapipe](https://github.com/google/mediapipe)

