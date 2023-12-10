# pip install face_alignment
# pip install mediapipe
# pip install dlib
# wget https://huggingface.co/spaces/asdasdasdasd/Face-forgery-detection/resolve/ccfc24642e0210d4d885bc7b3dbc9a68ed948ad6/shape_predictor_68_face_landmarks.dat



import cv2
import dlib
import face_alignment
import mediapipe as mp
import numpy as np
import os

# Initialize mediapipe face detection utility.
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Initialize face alignment from face-alignment library
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False, face_detector='sfd', verbose=True)

# Define font parameters for titles
font = cv2.FONT_HERSHEY_DUPLEX
font_scale = 0.5
font_color = (0, 0, 0)
font_thickness = 1

# Function to perform face alignment using Dlib
def dlib_alignment_68_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]  # Using the first detected face
        shape = predictor(gray, face)
        for i in range(68):
            center = (shape.part(i).x, shape.part(i).y)
            cv2.circle(image, center, 1, (255, 0, 0),1)
        return image, True
    return image, False

def face_alignment_68_points_fa(image):
    landmarks = fa.get_landmarks(image)
    if landmarks is not None and len(landmarks[0]) == 68:
        for landmark in landmarks[0]:
            center = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, center, 1, (0, 0, 255), 1)
    return image

def face_alignment_68_points_mp(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(landmarks.landmark):
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
    return image
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing file: {file_name}")
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, f"result_{file_name}")

            image = cv2.imread(input_path)

            image_fa = face_alignment_68_points_fa(image.copy())
            image_mp = face_alignment_68_points_mp(image.copy())
            image_dlib, face_detected = dlib_alignment_68_points(image.copy())

            titles = ["1adrianb", "mediapipe", "dlib No face detect" if not face_detected else "dlib"]
            images = [image_fa, image_mp, image_dlib]

            # Resize images to same height for consistent display
            max_height = max(image.shape[0], max(img.shape[0] for img in images)) + 20
            total_width = sum(img.shape[1] for img in images)
            canvas_height = max_height + image.shape[0] + 20
            canvas_width = max(image.shape[1], total_width)

            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            x_offset = (canvas_width - image.shape[1]) // 2
            canvas[:image.shape[0], x_offset:x_offset + image.shape[1]] = image

            # Place the aligned images and titles at the bottom of the canvas
            x_offset = 0
            for idx, (aligned_image, title) in enumerate(zip(images, titles)):
                canvas[max_height:max_height+aligned_image.shape[0], x_offset:x_offset+aligned_image.shape[1]] = aligned_image
                cv2.putText(canvas, title, (x_offset + 10, max_height - 10), font, font_scale, font_color, font_thickness)
                x_offset += aligned_image.shape[1]

            # Append a title for the original image at the top
            cv2.putText(canvas, "Original Image", (x_offset//3, 20), font, font_scale, font_color, font_thickness)

            cv2.imwrite(output_path, canvas)
            print(f"Processed and saved to: {output_path}")

# Initialize the input and output folders
input_folder_path = "images"
output_folder_path = "results_combined"

# Start processing the images
process_images(input_folder_path, output_folder_path)