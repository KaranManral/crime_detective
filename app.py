import os
from flask import Flask,jsonify,request,render_template,Response,flash
import cv2
from fer import FER
import mediapipe as mp
from flask_cors import CORS
from face_detection_model.model import train_model,add_face,delete_face,run_model
from live_crime_alert_model.project import detect_emergency_hand_signal,detect_emotions,detect_weapon,get_current_time,get_location,send_sms_alert

app = Flask(__name__,static_url_path='',static_folder=f"{os.getcwd()}/crime_detective_client/build",template_folder=f"{os.getcwd()}/crime_detective_client/build")
# app.secret_key = 'your_secret_key'  # Needed for flash messages
CORS(app)

camera = None
streaming = False
frame_count = 0
frame_skip = 2

net = None
output_layers = []
classes=[]
hands=[]
mp_hands=[]

known_face_names=[]
known_face_encodings=[]

def generate_frames(detect_hand_signal_flag,detect_weapon_flag,detect_face_expression_flag,detect_face_flag):
    global camera,streaming,net,output_layers,classes,frame_count,frame_skip,mp_hands,hands,emotion_detector,known_face_names,known_face_encodings
    print(streaming)
    while streaming:
        success,frame=camera.read()
        if not success:
            break
        else:
            
            if frame_count % frame_skip == 0:
                if(detect_weapon_flag):
                    weapon_detected, weapon_label = detect_weapon(frame=frame,net=net,output_layers=output_layers,classes=classes)
                    
                if(detect_face_expression_flag):
                    abnormal_expression_detected,neutral_expression_detected = detect_emotions(frame,emotion_detector=emotion_detector)
                
                if(detect_weapon_flag and detect_face_expression_flag):
                    # Send alert if neutral expression and weapon detected
                    if weapon_detected and neutral_expression_detected:
                        current_time = get_current_time()
                        location = get_location()
                        alert_message = f"Threat Detected \nTime: {current_time}\nLocation: {location}"
                        send_sms_alert(alert_message)
                        print(alert_message)

                    # Send alert if abnormal expression and weapon detected
                    if weapon_detected and abnormal_expression_detected:
                        current_time = get_current_time()
                        location = get_location()
                        alert_message = f"Threat Detected \nTime: {current_time}\nLocation: {location}"
                        send_sms_alert(alert_message)
                        print(alert_message)

                
                if(detect_hand_signal_flag):
                    emergency_hand_signal_detected = detect_emergency_hand_signal(frame=frame,hands=hands,mp_hands=mp_hands)
                    
                    # Send alert if hand gesture detected
                    if emergency_hand_signal_detected:
                        current_time = get_current_time()
                        location = get_location()
                        alert_message = f"Help Needed!\nTime: {current_time}\nLocation: {location}"
                        send_sms_alert(alert_message)
                        print(alert_message)


                if(detect_face_flag):
                    run_model(frame=frame,known_face_encodings=known_face_encodings,known_face_names=known_face_names)

            frame_count +=1


             # Draw a red dot at the top-left corner (5x5 pixels)
            dot_position = (10, 10)  # Position for the dot
            cv2.circle(frame, dot_position, 3, (0, 0, 255), -1)  # Draw the red dot

            # Add text next to the dot
            text_position = (15, 15)  # Position for the text
            cv2.putText(frame, 'Live', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            ret,buffer=cv2.imencode('.webp', frame, [cv2.IMWRITE_WEBP_QUALITY, 80])
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Preload libraries 
def load_libraries():
    global net, output_layers, classes,emotion_detector,mp_hands,hands,known_face_encodings,known_face_names
    # Load YOLO model for weapon detection
    net = cv2.dnn.readNet(f"{os.getcwd()}/live_crime_alert_model/yolov3.weights", f"{os.getcwd()}/live_crime_alert_model/yolov3.cfg")
    
    # Set preferable backend (optional GPU acceleration)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    except:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Load COCO class labels
    with open(f"{os.getcwd()}/live_crime_alert_model/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Load YOLO layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load FER emotion detection model
    emotion_detector = FER(mtcnn=True)

    # MediaPipe hands for gesture detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    known_face_encodings, known_face_names = train_model()  # Load model once at start

# Serve index.html for all non-API routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # Check if the path starts with "api", which means it's an API call
    if path.startswith('api'):
        return "API route not found", 404
    # For any other route, serve index.html and let React Router handle it
    return render_template('index.html')

@app.route('/detect_criminal')
def catch_detect_criminal():
    return render_template('index.html')

@app.route('/start-camera', methods=['POST'])
def start_camera():
    global camera,streaming
    if not streaming:
        camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
        if not camera.isOpened():
            return jsonify({"status": "error", "message": "Camera could not be opened."}), 500
        streaming = True
        return jsonify({"status": "started"})
    return jsonify({"status": "already started"})

@app.route('/stop-camera', methods=['POST'])
def stop_camera():
    global camera, streaming
    if streaming:
        streaming = False
        camera.release()
        camera=None
    return jsonify({"status": "stopped"})

@app.route('/detect', methods=['GET'])
def detect():
    global streaming
    # if not streaming:
        # return Response(status=403)
    return Response(generate_frames(detect_face_expression_flag=True,detect_hand_signal_flag=True,detect_weapon_flag=True,detect_face_flag=False),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/api/get_map')
# def get_map():
#     global geoData
#     print("Fetch Started")
    
#     return geoData.to_json()

# @app.route('/api/get_crime_data')
# def get_crime_data():
#     global latest_crime_data
#     print("Fetch Started")
    
#     return jsonify({"data":latest_crime_data})

@app.route('/api/add_face', methods=['POST'])
def add():
    global known_face_encodings,known_face_names
    name = request.form['name'].strip()
    criminal_id = request.form['criminal_id'].strip()
    user_folder = f'criminal_faces/{criminal_id}_{name}'

    if not os.path.isdir(user_folder):
        os.makedirs(user_folder)

    msg = add_face(name=name,criminal_id=criminal_id,files=request.files,user_folder=user_folder)

    # known_face_encodings, known_face_names = train_model()

    return jsonify({"message": msg})

    # flash(msg)

@app.route('/api/remove_face', methods=['POST'])
def delete():
    global known_face_encodings,known_face_names
    name = request.form['name'].strip()
    criminal_id = request.form['criminal_id'].strip()
    user_folder = f'criminal_faces/{criminal_id}_{name}'
    known_face_names = []
    known_face_encodings = []
    msg = delete_face(name=name,criminal_id=criminal_id,user_folder=user_folder)
    
    return jsonify({"message": msg})

    # flash(msg)

@app.route('/api/detect_criminal', methods=['GET'])
def detect_criminal():
    global streaming
    # if not streaming:
        # return Response(status=403)
    return Response(generate_frames(detect_face_expression_flag=False,detect_hand_signal_flag=False,detect_weapon_flag=False,detect_face_flag=True),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_libraries()
    app.run(debug=True)