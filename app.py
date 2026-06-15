import av
import cv2
import math
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="NeuroNav-CV Demo", layout="wide")

st.title("NeuroNav-CV: Markerless Head Tracking Demo")
st.write(
    "Early proof-of-concept using a standard webcam to detect facial landmarks, "
    "generate a live facial mesh, and estimate head pose without physical markers."
)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

LANDMARK_IDS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

def rotation_matrix_to_euler_angles(rotation_matrix):
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        pitch = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = math.atan2(-rotation_matrix[2, 0], sy)
        roll = 0

    return np.degrees([pitch, yaw, roll])

class NeuroNavVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
            )

            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style(),
            )

            image_points = []
            for key in ["nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"]:
                lm = face_landmarks.landmark[LANDMARK_IDS[key]]
                x, y = int(lm.x * w), int(lm.y * h)
                image_points.append((x, y))
                cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

            image_points = np.array(image_points, dtype=np.float64)

            focal_length = w
            camera_matrix = np.array([
                [focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]
            ], dtype=np.float64)

            dist_coeffs = np.zeros((4, 1))

            success_pnp, rotation_vector, translation_vector = cv2.solvePnP(
                FACE_3D_MODEL_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success_pnp:
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                pitch, yaw, roll = rotation_matrix_to_euler_angles(rotation_matrix)

                nose_2d = tuple(image_points[0].astype(int))
                nose_projection, _ = cv2.projectPoints(
                    np.array([(0.0, 0.0, 100.0)], dtype=np.float64),
                    rotation_vector,
                    translation_vector,
                    camera_matrix,
                    dist_coeffs
                )

                nose_direction = tuple(nose_projection[0][0].astype(int))
                cv2.line(img, nose_2d, nose_direction, (255, 0, 0), 3)

                cv2.putText(img, "TRACKING: LOCKED", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                cv2.putText(img, f"Pitch: {pitch:.1f}", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(img, f"Yaw:   {yaw:.1f}", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(img, f"Roll:  {roll:.1f}", (20, 145),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                cv2.putText(img, f"X: {translation_vector[0][0]:.1f}", (20, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(img, f"Y: {translation_vector[1][0]:.1f}", (20, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(img, f"Z: {translation_vector[2][0]:.1f}", (20, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        else:
            cv2.putText(img, "TRACKING: SEARCHING", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.info(
    "Click START below and allow webcam access. Move your head left/right/up/down. "
    "This is not yet MRI/DICOM registration; it is the first markerless tracking proof-of-concept."
)

webrtc_streamer(
    key="neuronav-cv-demo",
    video_processor_factory=NeuroNavVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)