from flask import Flask, request, jsonify, send_file
from PIL import Image
from flask_cors import CORS
import os
import time
import face_recognition
import numpy as np
from werkzeug.utils import secure_filename
import re
import json


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

known_encodings = []
known_images = []

def calculate_face_distance(known_images, known_encodings, unknown_img_path, cutoff=0.5, num_result=4):
    image_to_test = face_recognition.load_image_file(unknown_img_path)
    image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
    
    face_distances = face_recognition.face_distance(known_encodings, image_to_test_encoding)
    return (unknown_img_path, known_images[face_distances.argmin()])

def loadEncodings():
    encs = []
    actors = []
    with open(os.path.join(BASE_DIR, "encodings.txt"),'r') as fh:
        lines = fh.readlines()
        for line in lines:
            encs.append(np.array([float(num) for num in line.split()]))
    
    with open(os.path.join(BASE_DIR, "actors.txt"), 'r') as fh:
        lines = fh.readlines()
        for line in lines:
            actors.append(line[0:-1])
        
    return (encs, actors)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    username = request.form.get("username", "user")  # Default username if not provided

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Sanitize username (replace spaces & special characters)
    username = re.sub(r'\W+', '_', username)  

    # Generate unique filename with username + timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = secure_filename(f"{username}_{timestamp}.jpg")
    upload_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save original uploaded image
    file.save(upload_path)

    # Load encodings and find matching celebrity
    known_encodings, known_images = loadEncodings()

    # Handle cases where no face is detected
    image_to_test = face_recognition.load_image_file(upload_path)
    encodings = face_recognition.face_encodings(image_to_test)
    if not encodings: 
        return jsonify({"error": "No face detected in the uploaded image"}), 400
    image_to_test_encoding = encodings[0]

    matching_image = calculate_face_distance(known_images, known_encodings, upload_path)[1]
    with open(os.path.join(BASE_DIR, "urls.json"), 'r') as fh:
        urls = json.load(fh)
    return jsonify({"image_url": urls[matching_image[0:-4]]})

@app.route("/images/<filename>")
def get_processed_image(filename):
    """Serve the processed image to the frontend"""
    with open(os.path.join(BASE_DIR, "urls.json"), 'r') as fh:
        urls = json.load(fh)
    file_path = urls[filename[0:-4]] 
    if os.path.exists(file_path):
        return send_file(file_path, mimetype="image/jpeg")
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

