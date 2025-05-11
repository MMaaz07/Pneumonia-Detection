from flask import Flask, request, jsonify, send_from_directory, url_for,render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import os
import json

app = Flask(__name__, static_folder=os.path.join(os.getcwd(), "static"))
CORS(app)

import glob

def clear_static_folder():
    files = glob.glob('static/*')
    for f in files:
        os.remove(f)
    print("✅ Cleared old static files on startup.")

clear_static_folder()



# Ensure the "static/" folder exists
static_folder_path = os.path.abspath("static")
if not os.path.exists(static_folder_path):
    os.makedirs(static_folder_path)
    print(f"✅ Created static folder at: {static_folder_path}")


# ✅ Step 1: Load Model Correctly
def load_model():
    model = models.densenet169(pretrained=False)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 2)
    try:
        model.load_state_dict(torch.load("", map_location=torch.device('cpu'))) #add model.pt 
        print("✅ Model loaded successfully from model.pt")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    model.eval()
    return model

model = load_model()


from tensorflow.keras.models import load_model as load_tf_model_lib
from tensorflow.keras.preprocessing.image import img_to_array

def load_tf_model():
    try:
        tf_model = load_tf_model_lib(r"", compile=False) #add model.h5
        tf_model.make_predict_function()  # Required for thread safety
        print("✅ TensorFlow model loaded successfully")
        return tf_model
    except Exception as e:
        print(f"❌ Error loading TensorFlow model: {e}")
        raise

tf_model = load_tf_model()



# ✅ Step 2: Define Image Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def calculate_severity(probability, threshold=0.5):
    percentage = probability * 100
    if probability < threshold:
        return "Normal", 0
    elif percentage < 40:
        return "Mild", 30
    elif 40 <= percentage < 70:
        return "Moderate", 60
    else:
        return "Severe", 100

@app.route('/predict', methods=['POST'])
def predict():
    clear_static_folder()
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        processed_image = preprocess_tf_image(image)
        raw_pred = tf_model.predict(processed_image)
        
        if raw_pred.shape[-1] == 1:
            pneumonia_prob = float(raw_pred[0][0])
            normal_prob = 1 - pneumonia_prob
        else:
            normal_prob = float(raw_pred[0][0])
            pneumonia_prob = float(raw_pred[0][1])

        threshold = 0.5
        is_pneumonia = pneumonia_prob > threshold

        severity, progress = calculate_severity(pneumonia_prob)

        return jsonify({
            "probabilities": {
                "normal": normal_prob,
                "pneumonia": pneumonia_prob
            },
            "pneumonia_detected": f"Pneumonia Detected" if pneumonia_prob > 0.5 else "No Pneumonia",
            "severity": severity,
            "progress": progress
        })
    except Exception as predict_e:
        print(f"❌ Prediction Error: {predict_e}")
        return jsonify({"error": f"Prediction failed: {str(predict_e)}"}), 500






# ✅ Step 4: Serve Static Files
def serve_visualization(filename):
    file_path = os.path.join(static_folder_path, filename)
    if not os.path.exists(file_path):
        return jsonify({"success": False, "error": "File not found!"}), 404
    return send_from_directory(static_folder_path, filename)

# ✅ Step 5: Preprocess Image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor



def preprocess_tf_image(image_input):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")  # 📂 If path is given, open
    else:
        image = image_input.convert("RGB")  # 🖼️ If already an image, just ensure RGB

    input_shape = tf_model.input_shape[1:3]
    print(f"Model expects input shape: {input_shape}")

    image = image.resize(input_shape)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image





# ✅ Step 6: Generate Grad-CAM
def grad_cam(model, image_tensor, target_class=None):
    activations = None
    gradients = None
    
    def save_activations(module, input, output):
        nonlocal activations
        activations = output

    def save_gradients(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    last_conv_layer = model.features[-1]
    hook_activations = last_conv_layer.register_forward_hook(save_activations)
    hook_gradients = last_conv_layer.register_backward_hook(save_gradients)

    output = model(image_tensor)
    if target_class is None:
        target_class = torch.argmax(output, dim=1)
    
    model.zero_grad()
    target = output[0, target_class]
    target.backward()
    
    hook_activations.remove()
    hook_gradients.remove()
    
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    grad_cam_output = torch.sum(weights * activations, dim=1, keepdim=True)
    grad_cam_output = torch.relu(grad_cam_output)
    grad_cam_output = grad_cam_output.squeeze().cpu().detach().numpy()
    grad_cam_output = cv2.resize(grad_cam_output, (224, 224))
    grad_cam_output -= grad_cam_output.min()
    grad_cam_output /= grad_cam_output.max()
    
    return grad_cam_output, target_class.item()




# ✅ Step 7: Overlay Heatmap on Image
import datetime

def overlay_heatmap(image_path, heatmap):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    # Add timestamp to the visualization
    timestamp_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(superimposed_img, timestamp_text, (20, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    output_path = os.path.join("static", f"gradcam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return output_path







# ✅ Step 8: Flask Route to Handle Visualization
import base64

@app.route("/visualize", methods=["POST"])
def visualize():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    temp_image_path = os.path.join(static_folder_path, "temp_image.jpg")
    file.save(temp_image_path)

    # ✅ Step 1: Predict using TensorFlow model (REAL prediction)
    tf_image = preprocess_tf_image(temp_image_path)
    raw_pred = tf_model.predict(tf_image)

    if raw_pred.shape[-1] == 1:
        pneumonia_prob = float(raw_pred[0][0])
    else:
        pneumonia_prob = float(raw_pred[0][1])

    threshold = 0.5
    is_pneumonia = pneumonia_prob > threshold

    # ✅ Step 2: If Normal, return message directly
    if not is_pneumonia:
        return jsonify({"success": True, "message": "No signs of pneumonia!"})

    # ✅ Step 3: Otherwise, generate GradCAM using PyTorch model
    image_tensor = preprocess_image(temp_image_path)
    heatmap, _ = grad_cam(model, image_tensor)  # We don't trust grad_cam prediction anymore.

    heatmap_path = overlay_heatmap(temp_image_path, heatmap)
    return jsonify({"success": True, "image_url": url_for('static', filename=os.path.basename(heatmap_path))})







@app.route('/assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory("assets", filename)


# ✅ Step 9: Static File Serving
@app.route('/static/<path:filename>')
def serve_static_files(filename):
    static_folder_path = os.path.abspath("static")  # Get absolute path of static folder
    print(f"📂 Serving file: {filename} from {static_folder_path}")  # Debugging
    return send_from_directory(static_folder_path, filename, cache_timeout=0)


@app.route('/normal_message')
def normal_message():
    return render_template("normal_message.html", message="You're absolutely fine, No signs of pneumonia!")




# ✅ Step 10: Welcome Page
#@app.route('/')
#def welcome():
#    return "<html><body><h1>We're Live Now</h1></body></html>"


@app.route('/')
def index():
    clear_static_folder()
    return render_template("frontend.html")



if __name__ == "__main__":
    app.run(debug=True, port=5000)


"""
Author: mysgrade
Date: 2025-02-21
Description: This Flask app handles requests for the Pneumonia Detcetion web application.
"""
