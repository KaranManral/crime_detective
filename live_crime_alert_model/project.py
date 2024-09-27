import cv2
import numpy as np
import mediapipe as mp
from twilio.rest import Client
from datetime import datetime
import geocoder

# # Twilio credentials
# TWILIO_ACCOUNT_SID = 'ACa0432c2550d2e58bfc28c9dbc02ec323'
# TWILIO_AUTH_TOKEN = '1f6f5cc98b3ca406b3b37c5f64317582'
# TWILIO_PHONE_NUMBER = '+13343104461'
# ALERT_PHONE_NUMBER = '+918617229290'

# Twilio client
# client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
def detect_weapon(frame,net,output_layers,classes, confidence_threshold=0.3, nms_threshold=0.3):
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
def detect_emergency_hand_signal(frame,hands,mp_hands):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        if thumb_tip.x < index_mcp.x and thumb_tip.y > pinky_tip.y:
            return True
    return False


def detect_emotions(frame,emotion_detector):
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

    return abnormal_expression_detected,neutral_expression_detected

# # Main function for real-time expression, weapon, and hand gesture detection
# def crime_detection():
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # # Step 1: Detect facial expressions
#         # emotions = emotion_detector.detect_emotions(frame)
#         # abnormal_expression_detected = False
#         # neutral_expression_detected = False

#         # for emotion in emotions:
#         #     dominant_emotion = max(emotion['emotions'], key=emotion['emotions'].get)
#         #     emotion_score = emotion['emotions'][dominant_emotion]

#         #     if emotion['emotions']['angry'] > 0.5:
#         #         abnormal_expression_detected = True
#         #     if emotion['emotions']['neutral'] > 0.5:
#         #         neutral_expression_detected = True

#         #     x, y, w, h = emotion['box']
#         #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         #     cv2.putText(frame, f"{dominant_emotion.capitalize()} ({emotion_score:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#         # Step 2: Detect weapon
#         # weapon_detected, weapon_label = detect_weapon(frame)

#         # if weapon_detected:
#         #     cv2.putText(frame, f"Weapon Detected: {weapon_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         # # Step 3: Detect emergency hand gesture
#         # emergency_signal_detected = detect_emergency_hand_signal(frame)

#         # # Display the result
#         # cv2.imshow('Expression, Weapon, and Hand Gesture Detection', frame)

#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break

#     cap.release()
#     cv2.destroyAllWindows()

# # Running the program
# if __name__ == "__main__":
#     crime_detection()
