from mtcnn import MTCNN
import cv2

detector = MTCNN()

image = cv2.imread("multiple_face.png")

if image is None:
    raise FileNotFoundError("image.png not found or could not be loaded")

faces = detector.detect_faces(image)
print(f"Faces detected: {len(faces)}")

for i, face in enumerate(faces):
    # Bounding box
    x, y, w, h = face['box']
    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Landmarks
    keypoints = face['keypoints']

    for name, point in keypoints.items():
        cv2.circle(
            image,
            point,
            radius=3,
            color=(0, 255, 0),
            thickness=-1
        )

    # Optional: label face number
    cv2.putText(
        image,
        f"Face {i+1}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

# Save output
cv2.imwrite("output.png", image)
