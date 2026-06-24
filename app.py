"""
Soil Classification Backend
FastAPI app that loads a CNN model to classify soil images and recommend crops.
Deployed on Hugging Face Spaces via Docker.
"""

import os
import json
import zipfile
import base64
import io
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("soil-backend")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "model/soil_model007.keras")
DATA_PATH = os.environ.get("DATA_PATH", "data/data_core.csv")
IMG_SIZE = (224, 224)

# Classes in alphabetical order — default Keras behaviour when training
# with image_dataset_from_directory or ImageDataGenerator.
SOIL_CLASSES = ["Black", "Clayey", "Loamy", "Red", "Sandy"]

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------
model = None
crop_lookup: dict = {}

# ---------------------------------------------------------------------------
# Soil explanations per language
# ---------------------------------------------------------------------------
SOIL_EXPLANATIONS = {
    "Black": {
        "English": (
            "This soil appears to be Black soil (also known as Regur soil). "
            "It is rich in clay, iron, magnesium, and calcium. Black soil is "
            "highly moisture-retentive and ideal for cotton and other "
            "deep-rooted crops."
        ),
        "Hindi": (
            "यह मिट्टी काली मिट्टी (रेगुर मिट्टी) प्रतीत होती है। यह चिकनी मिट्टी, "
            "लोहा, मैग्नीशियम और कैल्शियम से समृद्ध है। काली मिट्टी नमी बनाए "
            "रखने में उत्कृष्ट है और कपास तथा गहरी जड़ वाली फसलों के लिए आदर्श है।"
        ),
        "Odia": (
            "ଏହା କଳା ମାଟି (ରେଗୁର ମାଟି) ବୋଲି ମନେ ହେଉଛି। ଏଥିରେ ଚିକିଟା ମାଟି, "
            "ଲୁହା, ମ୍ୟାଗ୍ନେସିୟମ ଏବଂ କ୍ୟାଲସିୟମ ଭରପୂର। କଳା ମାଟି ଆର୍ଦ୍ରତା ଧାରଣ "
            "କ୍ଷମତା ଅଧିକ ଏବଂ କପା ଓ ଗଭୀର ଚେର ଫସଲ ପାଇଁ ଉପଯୁକ୍ତ।"
        ),
    },
    "Clayey": {
        "English": (
            "This soil appears to be Clayey soil. It has very fine particles "
            "that hold water well but can become waterlogged. Clayey soil is "
            "nutrient-rich and suitable for crops that thrive in moist "
            "conditions like paddy and wheat."
        ),
        "Hindi": (
            "यह मिट्टी चिकनी मिट्टी प्रतीत होती है। इसमें बहुत महीन कण होते हैं "
            "जो पानी अच्छी तरह रोकते हैं लेकिन जलभराव हो सकता है। चिकनी मिट्टी "
            "पोषक तत्वों से भरपूर है और धान व गेहूं जैसी फसलों के लिए उपयुक्त है।"
        ),
        "Odia": (
            "ଏହା ଚିକିଟା ମାଟି ବୋଲି ମନେ ହେଉଛି। ଏଥିରେ ଅତି ସୂକ୍ଷ୍ମ କଣିକା ଥାଏ "
            "ଯାହା ଜଳ ଭଲ ଭାବରେ ଧରି ରଖେ କିନ୍ତୁ ଜଳମଗ୍ନ ହୋଇପାରେ। ଚିକିଟା ମାଟି ପୋଷକ "
            "ତତ୍ତ୍ୱରେ ସମୃଦ୍ଧ ଏବଂ ଧାନ ଓ ଗହମ ଭଳି ଫସଲ ପାଇଁ ଉପଯୁକ୍ତ।"
        ),
    },
    "Loamy": {
        "English": (
            "This soil appears to be Loamy soil. It is a balanced mix of "
            "sand, silt, and clay — considered the ideal agricultural soil. "
            "Loamy soil has excellent drainage, good moisture retention, and "
            "is rich in nutrients, supporting a wide variety of crops."
        ),
        "Hindi": (
            "यह मिट्टी दोमट मिट्टी प्रतीत होती है। यह रेत, गाद और चिकनी मिट्टी "
            "का संतुलित मिश्रण है — इसे आदर्श कृषि मिट्टी माना जाता है। दोमट मिट्टी "
            "में उत्कृष्ट जल निकास, अच्छी नमी धारण क्षमता और प्रचुर पोषक तत्व होते हैं।"
        ),
        "Odia": (
            "ଏହା ଦୋମାଟି ମାଟି ବୋଲି ମନେ ହେଉଛି। ଏହା ବାଲି, ପଙ୍କ ଏବଂ ଚିକିଟା ମାଟିର "
            "ଏକ ସନ୍ତୁଳିତ ମିଶ୍ରଣ — ଏହାକୁ ଆଦର୍ଶ କୃଷି ମାଟି ବୋଲି ମନା ଯାଏ। ଦୋମାଟି "
            "ମାଟିରେ ଉତ୍କୃଷ୍ଟ ନିଷ୍କାସନ, ଭଲ ଆର୍ଦ୍ରତା ଧାରଣ ଏବଂ ପ୍ରଚୁର ପୋଷକ ତତ୍ତ୍ୱ ଥାଏ।"
        ),
    },
    "Red": {
        "English": (
            "This soil appears to be Red soil. It gets its colour from iron "
            "oxide content. Red soil is typically well-drained, slightly "
            "acidic, and found in areas with moderate rainfall. It is "
            "suitable for groundnuts, millets, and tobacco with proper "
            "fertilisation."
        ),
        "Hindi": (
            "यह मिट्टी लाल मिट्टी प्रतीत होती है। इसका रंग लौह ऑक्साइड की मात्रा "
            "से आता है। लाल मिट्टी आम तौर पर अच्छी जल निकासी वाली, हल्की अम्लीय "
            "होती है। उचित उर्वरक के साथ मूंगफली, बाजरा और तम्बाकू जैसी फसलों के "
            "लिए उपयुक्त है।"
        ),
        "Odia": (
            "ଏହା ଲାଲ ମାଟି ବୋଲି ମନେ ହେଉଛି। ଏହାର ରଙ୍ଗ ଲୌହ ଅକ୍ସାଇଡ୍ ଯୋଗୁଁ "
            "ହୋଇଥାଏ। ଲାଲ ମାଟି ସାଧାରଣତଃ ଭଲ ନିଷ୍କାସନ ଥିବା, ସାମାନ୍ୟ ଅମ୍ଳୀୟ। "
            "ଉପଯୁକ୍ତ ସାର ସହ ବାଦାମ, ମିଲେଟ ଓ ତମାଖୁ ଭଳି ଫସଲ ପାଇଁ ଏହା ଉପଯୁକ୍ତ।"
        ),
    },
    "Sandy": {
        "English": (
            "This soil appears to be Sandy soil. It has large, loose "
            "particles that provide excellent drainage but poor moisture "
            "and nutrient retention. Sandy soil warms up quickly and is "
            "suitable for barley, millets, and maize that tolerate dry "
            "conditions."
        ),
        "Hindi": (
            "यह मिट्टी बलुई मिट्टी प्रतीत होती है। इसमें बड़े, ढीले कण होते हैं "
            "जो उत्कृष्ट जल निकास प्रदान करते हैं लेकिन नमी और पोषक तत्व कम "
            "बनाए रखते हैं। बलुई मिट्टी जौ, बाजरा और मक्का जैसी शुष्क फसलों के "
            "लिए उपयुक्त है।"
        ),
        "Odia": (
            "ଏହା ବାଲୁକା ମାଟି ବୋଲି ମନେ ହେଉଛି। ଏଥିରେ ବଡ଼, ଢିଲା କଣିକା ଥାଏ "
            "ଯାହା ଉତ୍କୃଷ୍ଟ ନିଷ୍କାସନ ଦିଏ କିନ୍ତୁ ଆର୍ଦ୍ରତା ଓ ପୋଷକ ତତ୍ତ୍ୱ ଧାରଣ "
            "କ୍ଷମତା କମ୍। ବାଲୁକା ମାଟି ଯବ, ମିଲେଟ ଓ ମକା ଭଳି ଶୁଷ୍କ ଫସଲ ପାଇଁ ଉପଯୁକ୍ତ।"
        ),
    },
}

# ---------------------------------------------------------------------------
# Pydantic schemas (match frontend types.ts)
# ---------------------------------------------------------------------------

class ClassifyRequest(BaseModel):
    imageBase64: str
    language: str = "English"


class ClassificationResult(BaseModel):
    soilType: str
    explanation: str
    recommendedCrops: List[str]


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _clean_model_config(obj):
    """Recursively strip `quantization_config` from a Keras config dict.

    This fixes the version-mismatch error:
        TypeError: Unrecognized keyword arguments passed to Dense:
        {'quantization_config': None}
    """
    if isinstance(obj, dict):
        obj.pop("quantization_config", None)
        for value in obj.values():
            _clean_model_config(value)
    elif isinstance(obj, list):
        for item in obj:
            _clean_model_config(item)
    return obj


def _load_model(path: str):
    """Load a .keras model with a fallback that cleans incompatible config."""
    import tensorflow as tf

    # --- Attempt 1: direct load ---
    try:
        m = tf.keras.models.load_model(path)
        logger.info("Model loaded successfully (direct).")
        return m
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise
        logger.warning(
            "Direct load failed due to quantization_config. "
            "Cleaning config and retrying…"
        )

    # --- Attempt 2: strip quantization_config from the zip ---
    cleaned_path = path + ".cleaned.keras"
    with zipfile.ZipFile(path, "r") as zin, \
         zipfile.ZipFile(cleaned_path, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(data)
                config = _clean_model_config(config)
                data = json.dumps(config).encode("utf-8")
            zout.writestr(item, data)

    m = tf.keras.models.load_model(cleaned_path)
    logger.info("Model loaded successfully (cleaned config).")

    # Clean up temp file
    try:
        os.remove(cleaned_path)
    except OSError:
        pass

    return m


def _build_crop_lookup(csv_path: str) -> dict:
    """Build a mapping: soil_type → list of unique crop types from the CSV."""
    df = pd.read_csv(csv_path)
    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()
    lookup = {}
    for soil in df["Soil Type"].unique():
        crops = (
            df.loc[df["Soil Type"] == soil, "Crop Type"]
            .dropna()
            .unique()
            .tolist()
        )
        lookup[soil] = sorted(crops)
    return lookup


# ---------------------------------------------------------------------------
# Application lifespan — load model & data once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, crop_lookup

    logger.info("Loading CNN model from %s …", MODEL_PATH)
    model = _load_model(MODEL_PATH)
    logger.info("Model input shape : %s", model.input_shape)
    logger.info("Model output shape: %s", model.output_shape)

    logger.info("Loading crop data from %s …", DATA_PATH)
    crop_lookup = _build_crop_lookup(DATA_PATH)
    logger.info("Crop lookup built for soil types: %s", list(crop_lookup.keys()))

    yield  # application runs here

    logger.info("Shutting down — releasing model.")
    model = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Soil Classification API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Wasmer-hosted frontend (and any other origin during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper — decode & preprocess image
# ---------------------------------------------------------------------------

def _preprocess_image(base64_str: str) -> np.ndarray:
    """Decode a data-URI / raw base64 string into a (1, 224, 224, 3) array."""
    # Strip the data URI prefix if present  (e.g. "data:image/jpeg;base64,")
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    arr = np.array(img, dtype=np.float32) / 255.0  # normalise to [0, 1]
    return np.expand_dims(arr, axis=0)  # add batch dimension


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    """Health-check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "soil_classes": SOIL_CLASSES,
    }


@app.post("/api/classify", response_model=ClassificationResult)
async def classify_soil(payload: ClassifyRequest):
    """Classify a soil image and return soil type + crop recommendations."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # 1. Preprocess image
    try:
        img_array = _preprocess_image(payload.imageBase64)
    except Exception as exc:
        logger.error("Image preprocessing failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid image data.")

    # 2. Run prediction
    predictions = model.predict(img_array, verbose=0)
    class_idx = int(np.argmax(predictions[0]))
    soil_type = SOIL_CLASSES[class_idx]

    logger.info(
        "Prediction: %s (confidence %.2f%%)",
        soil_type,
        float(predictions[0][class_idx]) * 100,
    )

    # 3. Look up recommended crops from CSV
    recommended_crops = crop_lookup.get(soil_type, [])

    # 4. Get explanation in the requested language
    lang = payload.language if payload.language in ("English", "Hindi", "Odia") else "English"
    explanation = SOIL_EXPLANATIONS.get(soil_type, {}).get(lang, "")

    return ClassificationResult(
        soilType=soil_type,
        explanation=explanation,
        recommendedCrops=recommended_crops,
    )
