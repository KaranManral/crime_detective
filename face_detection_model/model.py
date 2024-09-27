import os
import cv2
import face_recognition
import numpy as np
import shutil
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

# Create the 'criminal_faces' directory if it doesn't exist
if not os.path.isdir('criminal_faces'):
    os.makedirs('criminal_faces')

# Add new face
def add_face(name,criminal_id,files,user_folder):

    # Save the uploaded images
    if 'images' not in files:
        return "No images uploaded."

    files = files.getlist('images')
    for file in files:
        if file.filename:
            filepath = os.path.join(user_folder, file.filename)
            file.save(filepath)

    return f'Added {name} successfully!'

def delete_face(name,criminal_id,user_folder):

    if os.path.isdir(user_folder):
        shutil.rmtree(user_folder)
        return f'Deleted {name} successfully!'
    else:
        return f'Criminal ID {criminal_id} with name {name} not found!'

# Function to train the face recognition model
def train_model():
    known_face_encodings = []
    known_face_names = []

    # Iterate through folders in 'criminal_faces' directory
    for user in os.listdir('criminal_faces'):
        user_folder = os.path.join('criminal_faces', user)
        for imgname in os.listdir(user_folder):
            img_path = os.path.join(user_folder, imgname)
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(user)

    return known_face_encodings, known_face_names

def run_model(frame,known_face_encodings,known_face_names):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            current_time = get_current_time()
            location = get_location()
            alert_message = f"Criminal Detected!\nName: {name}\nTime: {current_time}\nLocation: {location}"
            # send_sms_alert(alert_message)
            print(alert_message)

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 20), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 20), 2)