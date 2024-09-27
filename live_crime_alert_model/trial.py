import cv2
import numpy as np
from fer import FER
from twilio.rest import Client
from datetime import datetime
import geocoder
import mediapipe as mp

# Load FER emotion detection model
emotion_detector = FER(mtcnn=True)

# Load YOLO model for weapon detection
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# # Twilio credentials
# TWILIO_ACCOUNT_SID = 'ACac88f07cc4c8367a1f5e7a3d1f86c94d'
# TWILIO_AUTH_TOKEN = '169e002c6c68338c3b8d54cba3186589'
# TWILIO_PHONE_NUMBER = '+13343102046'
# ALERT_PHONE_NUMBER = '+919026172491'

# Twilio client
# client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# MediaPipe hands for gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to get the current time
def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Function to get the user's location based on IP address
def get_location():
    g = geocoder.ip('me')
    return f"{g.city}, {g.state}, {g.country}" if g.ok else "Unknown Location"

# Function to send SMS alert using Twilio
def send_sms_alert(message):
    try:
        # message = client.messages.create(
        #     body=message,
        #     from_=TWILIO_PHONE_NUMBER,
        #     to=ALERT_PHONE_NUMBER
        # )
        # print(f"Alert sent: {message.sid}")
        print(f"Alert sent:")
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Function to detect weapons using YOLO
def detect_weapon(frame, confidence_threshold=0.3, nms_threshold=0.3):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    weapon_keywords = {"knife", "gun", "rifle", "pistol"}
    weapon_detected = False
    weapon_label = ""

    if len(indices) > 0:
        for i in indices.flatten():
            label = classes[class_ids[i]]
            if any(keyword in label.lower() for keyword in weapon_keywords):
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                weapon_label = label.capitalize()
                cv2.putText(frame, f"Weapon: {weapon_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                weapon_detected = True

    return weapon_detected, weapon_label

# Function to detect emergency hand gesture
def detect_emergency_hand_signal(frame):
    emergency_signal_detected = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            if thumb_tip.x < index_mcp.x and thumb_tip.y > pinky_tip.y:
                emergency_signal_detected = True
                h, w, _ = frame.shape
                x_min = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * w))
                x_max = int(max(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * w))
                y_min = int(min(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h,
                                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * h))
                y_max = int(max(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h,
                                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * h))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, "Emergency Signal Detected", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return emergency_signal_detected

# Main function for real-time expression, weapon, and hand gesture detection
def crime_detection():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect facial expressions
        emotions = emotion_detector.detect_emotions(frame)
        abnormal_expression_detected = False
        neutral_expression_detected = False

        for emotion in emotions:
            dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
            emotion_score = emotion['emotions'][dominant_emotion]

            if emotion['emotions']['angry'] > 0.5:
                abnormal_expression_detected = True
            if emotion['emotions']['neutral'] > 0.5:
                neutral_expression_detected = True

            x, y, w, h = emotion['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant_emotion.capitalize()} ({emotion_score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Step 2: Detect weapon
        weapon_detected, weapon_label = detect_weapon(frame)

        if weapon_detected:
            cv2.putText(frame, f"Weapon Detected: {weapon_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Step 3: Detect emergency hand gesture
        emergency_signal_detected = detect_emergency_hand_signal(frame)

        # Send alert if abnormal expression and weapon detected
        if abnormal_expression_detected and weapon_detected:
            current_time = get_current_time()
            location = get_location()
            alert_message = f"Threat Detected \nTime: {current_time}\nLocation: {location}"
            send_sms_alert(alert_message)

        # Send alert if neutral expression and weapon detected
        if neutral_expression_detected and weapon_detected:
            current_time = get_current_time()
            location = get_location()
            alert_message = f"Threat Detected \nTime: {current_time}\nLocation: {location}"
            send_sms_alert(alert_message)

        # Send alert if hand gesture detected
        if emergency_signal_detected:
            current_time = get_current_time()
            location = get_location()
            alert_message = f"Help Needed!\nTime: {current_time}\nLocation: {location}"
            send_sms_alert(alert_message)

        # Display the result
        cv2.imshow('Expression, Weapon, and Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Running the program
if __name__ == "__main__":
    crime_detection()
