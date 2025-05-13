
import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# Poner aquí el nombre del modelo para la detección de señales.
model = tf.keras.models.load_model("best_model")

nombres_senales = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)', 
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)', 
    'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection', 
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 
    'No entry', 'General caution', 'Dangerous curve left', 'Dangerous curve right', 
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 
    'Wild animals crossing', 'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 
    'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons'
]

# Para deteccion mas estable.
prediction_history = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue

        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / h
        extent = float(area) / (w * h)

        shape = "undefined"
        if len(approx) == 3:
            shape = ""
        elif len(approx) == 4:
            shape = ""
        elif len(approx) > 5:
            circularity = 4 * np.pi * (area / (cv2.arcLength(cnt, True) ** 2 + 1e-5))
            if 0.7 < circularity <= 1.3:
                shape = ""

        if shape != "undefined":
            candidates.append((x, y, w, h, shape))

    # Candidato mas grande
    if candidates:
        x, y, w, h, shape = max(candidates, key=lambda b: b[2] * b[3])
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output, shape.upper(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        roi = gray[y:y + h, x:x + w]

        try:
            h_roi, w_roi = roi.shape
            scale = 30 / max(h_roi, w_roi)
            resized = cv2.resize(roi, (int(w_roi * scale), int(h_roi * scale)))

            top = (30 - resized.shape[0]) // 2
            bottom = 30 - resized.shape[0] - top
            left = (30 - resized.shape[1]) // 2
            right = 30 - resized.shape[1] - left

            roi_padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                            borderType=cv2.BORDER_CONSTANT, value=0)

            roi_padded = cv2.GaussianBlur(roi_padded, (3, 3), 0)
            roi_padded = roi_padded[..., np.newaxis]

            roi_input = np.expand_dims(roi_padded / 255.0, axis=0)

            prediction = model.predict(roi_input, verbose=0)
            confidence = np.max(prediction)
            class_id = np.argmax(prediction)

            if confidence >= 0.90:
                prediction_history.append(class_id)

                if len(prediction_history) == prediction_history.maxlen:
                    most_common = Counter(prediction_history).most_common(1)[0]
                    if most_common[1] >= 3:  # Si aparece en las ultimas 3 frames
                        label = nombres_senales[most_common[0]]
                        cv2.putText(output, f"{label} ({confidence:.2f})", (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        debug_resized = cv2.resize(roi_padded, (120, 120), interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("CNN Input (30x30)", debug_resized)

        except Exception as e:
            print("Error during prediction:", e)

    cv2.imshow("Traffic Sign Detector", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
