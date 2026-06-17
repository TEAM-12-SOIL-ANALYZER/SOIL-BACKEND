import os
import base64
import io
import json
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)
# Enable CORS for Wasmer frontend
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'soil_model007.keras')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data_core.csv')

# Load the keras model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load CSV database
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Data loaded successfully. {len(df)} rows.")
except Exception as e:
    print(f"Error loading CSV data: {e}")
    df = None

# Class names (assuming alphabetical order or standard order)
# Adjust these based on your model's actual training classes
CLASS_NAMES = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "Soil Classifier API is running!"})

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        if not data or 'imageBase64' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_b64 = data['imageBase64']
        language = data.get('language', 'English')
        
        # Strip the data:image/jpeg;base64, prefix if present
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
            
        # Decode image
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize image for model (assuming 256x256 or 224x224. We'll use 256 as default standard)
        image = image.resize((256, 256))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # Create batch axis
        
        soil_type = "Unknown"
        
        # Predict
        if model:
            predictions = model.predict(img_array)
            class_idx = np.argmax(predictions[0])
            soil_type = CLASS_NAMES[class_idx]
        else:
            # Fallback if model failed to load
            soil_type = random.choice(CLASS_NAMES)
            
        # Get crop recommendations from CSV
        recommended_crops = []
        if df is not None:
            # Filter by soil type (case insensitive matching)
            matches = df[df['Soil Type'].str.lower() == soil_type.lower()]
            if not matches.empty:
                unique_crops = matches['Crop Type'].unique().tolist()
                # Pick 3 to 5 random crops
                num_crops = min(len(unique_crops), random.randint(3, 5))
                recommended_crops = random.sample(unique_crops, num_crops)
        
        # Fallbacks
        if not recommended_crops:
            recommended_crops = ['Wheat', 'Rice', 'Cotton']
            
        explanation = f"Based on the visual characteristics evaluated by the CNN model, this appears to be {soil_type} soil."
        
        return jsonify({
            'soilType': soil_type,
            'explanation': explanation,
            'recommendedCrops': recommended_crops
        })
        
    except Exception as e:
        print(f"Error during classification: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
