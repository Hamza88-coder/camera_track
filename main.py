import time
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

# Charger le modèle YOLOv8-Face
model = YOLO("face_yolov8s.pt")  # Mets le bon chemin vers ton modèle

# Charger Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5)

# Dossier pour surveiller les images
image_path = "image.png"  # Modifie selon ton besoin

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Erreur: Impossible de charger l'image.")
        return

    img_h, img_w, _ = image.shape

    # Détection des visages avec YOLO
    results = model(image)
    total_looking, total_detected = 0, 0

    for result in results:
        total_detected = len(results)
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            face_roi = image[y1:y2, x1:x2]

            # Conversion en RGB pour MediaPipe
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_mesh_results = face_mesh.process(face_rgb)

            if face_mesh_results.multi_face_landmarks:
                for face_landmarks in face_mesh_results.multi_face_landmarks:
                    face_2d, face_3d = [], []

                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx in [33, 263, 1, 61, 291, 199]:
                            x, y = int(lm.x * (x2 - x1)) + x1, int(lm.y * (y2 - y1)) + y1
                            face_2d.append([x, y])
                            face_3d.append([x, y, lm.z])

                    face_2d, face_3d = np.array(face_2d, dtype=np.float64), np.array(face_3d, dtype=np.float64)
                    focal_length = 1 * img_w
                    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    if success:
                        rmat, _ = cv2.Rodrigues(rot_vec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                        x, y, z = angles[0] * 360, angles[1] * 360, angles[2] * 360

                        # Déterminer la direction du regard
                        direction = "looking at the camera"
                        if y < -30 or y > 30 or x < -30 or x > 30:
                            direction = "not looking"
                        else:
                            total_looking += 1

                        # Dessiner sur l'image
                        color = (0, 255, 0) if direction == "looking at the camera" else (0, 0, 255)
                        cv2.putText(image, direction, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Calcul du ratio
    ratio = (total_looking / total_detected) * 100 if total_detected > 0 else 0
    cv2.putText(image, f"Ratio: {ratio:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Sauvegarde du résultat
    # Générer un timestamp pour le nom du fichier
    timestamp = int(time.time())

# Utiliser un raw string (r"") pour éviter les problèmes d'échappement
    output_path = r"C:\Users\HP\OneDrive\Desktop\gaze_track\outputs\image_{}.png".format(timestamp)

# Sauvegarde de l'image
    cv2.imwrite(output_path, image)

    print(f"Image sauvegardée : {output_path}")
    print(f"Image traitée : Ratio {ratio:.2f}%")

if __name__ == "__main__":
    while True:
        process_image(image_path)
        time.sleep(10)  # Attendre 10 secondes avant de recommencer
