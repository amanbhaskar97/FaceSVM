from flask import Flask, request, jsonify, send_from_directory, render_template
import numpy as np
import base64
import cv2
import json
import joblib
import os
import traceback
import pywt

app = Flask(__name__, static_folder='.', template_folder='.')

# Global variables for model and class mapping
__class_name_to_number = {}
__class_number_to_name = {}
__model = None

# Function to load saved model and artifacts
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    artifacts_path = "./artifacts/"
    if not os.path.exists(artifacts_path):
        artifacts_path = "../artifacts/"
    
    # Load the class dictionary (name to number and vice versa)
    with open(os.path.join(artifacts_path, "class_dictionary.json"), "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        # Load the pre-trained model
        with open(os.path.join(artifacts_path, 'saved_model.pkl'), 'rb') as f:
            __model = joblib.load(f)
    
    print("loading saved artifacts...done")

# Function to perform wavelet transformation (used in image preprocessing)
def w2d(img, mode='haar', level=1):
    imArray = img
    # Convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    # Convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # Compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H

# Helper function to extract face region with two eyes detected
def get_opencv_path():
    possible_paths = [
        './opencv/haarcascades/',
        './haarcascades/',
        '../opencv/haarcascades/',
        '../haarcascades/',
        cv2.data.haarcascades
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'haarcascade_frontalface_default.xml')):
            return path
    
    raise FileNotFoundError("Could not find OpenCV haarcascades directory")

def get_cv2_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    opencv_path = get_opencv_path()
    face_cascade = cv2.CascadeClassifier(os.path.join(opencv_path, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(opencv_path, 'haarcascade_eye.xml'))

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    if len(cropped_faces) == 0:
        return [img]  # Return the original image if no face is detected
        
    return cropped_faces

# Function to classify image based on the model
def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    result = []
    
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32*32*3, 1), scalled_img_har.reshape(32*32, 1)))

        len_image_array = 32*32*3 + 32*32
        final = combined_img.reshape(1, len_image_array).astype(float)
        
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

# Flask route to serve the static HTML page
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    if path == "" or path == "/":
        return render_template('index.html')  # Serve HTML from templates
    try:
        return send_from_directory(app.static_folder, path)  # Serve other static files
    except FileNotFoundError:
        return render_template('index.html')  # Fallback to index.html if file not found

# Flask route for image classification
@app.route('/classify_image', methods=['POST'])
def classify_image_endpoint():
    try:
        # Check if there's form data
        if not request.form:
            print("No form data received")
            return jsonify({'error': 'No form data received'}), 400
            
        # Check if image_data is in the form
        if 'image_data' not in request.form:
            print("Missing 'image_data' in form")
            return jsonify({'error': "Missing 'image_data' in request"}), 400
        
        image_data = request.form['image_data']
        
        # Validate image_data format
        if not image_data or not image_data.startswith('data:image'):
            print("Invalid image data format")
            return jsonify({'error': 'Invalid image data format. Must be a base64 encoded image.'}), 400
        
        result = classify_image(image_data)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Test route to check if the server is running
@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running'}), 200

# Run the Flask application
if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    load_saved_artifacts()  # Load model and artifacts before starting the server
    app.run(port=5000, debug=True)
