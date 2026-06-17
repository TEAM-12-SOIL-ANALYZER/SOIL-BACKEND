# 🌾 Soil & Crop Classifier — Backend

A Flask-based REST API that classifies soil types from images using a trained CNN (TensorFlow/Keras) model and recommends suitable crops from a curated dataset.

---

## ✨ Features

- **🧠 CNN Soil Classification** — Uses a TensorFlow Keras model (`soil_model007.keras`) to classify soil into 5 types
- **🌾 Crop Recommendation Engine** — Matches detected soil type against a CSV database to suggest optimal crops
- **📡 RESTful API** — Simple JSON-based API with a single classification endpoint
- **🔒 CORS Enabled** — Pre-configured for cross-origin requests from any frontend
- **🐳 Docker Ready** — Includes a Dockerfile for containerized deployments
- **☁️ Cloud Deployable** — Pre-configured for Hugging Face Spaces, Railway, and Heroku

---

## 🛠️ Tech Stack

| Layer         | Technology                                                     |
| ------------- | -------------------------------------------------------------- |
| Framework     | [Flask 3.0](https://flask.palletsprojects.com/)                |
| ML Runtime    | [TensorFlow 2.16](https://www.tensorflow.org/)                 |
| Image Processing | [Pillow 10](https://pillow.readthedocs.io/)                |
| Data Handling | [Pandas 2.1](https://pandas.pydata.org/) + [NumPy 1.26](https://numpy.org/) |
| WSGI Server   | [Gunicorn 21](https://gunicorn.org/)                           |
| CORS          | [Flask-CORS 4.0](https://flask-cors.readthedocs.io/)          |

---

## 🗂️ Project Structure

```
backend/
├── main.py              # Flask app — API routes, model inference, crop lookup
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker image (Python 3.10-slim)
├── Procfile             # Heroku / PaaS process file
├── railway.toml         # Railway deployment config
├── .gitignore           # Git ignore rules
├── model/
│   └── soil_model007.keras   # Trained CNN model (~19 MB)
└── data/
    └── data_core.csv         # Soil-to-crop mapping dataset (~365 KB)
```

---

## 🧪 Soil Classification

### Supported Soil Types

The CNN model classifies images into one of **5 soil types**:

| #  | Soil Type | Description                                         |
| -- | --------- | --------------------------------------------------- |
| 1  | **Black**   | Dark, nutrient-rich soil (e.g., Regur / Black Cotton) |
| 2  | **Clayey**  | Heavy, water-retentive soil with fine particles       |
| 3  | **Loamy**   | Balanced soil — ideal for most crops                  |
| 4  | **Red**     | Iron-rich, acidic soil common in tropical regions     |
| 5  | **Sandy**   | Light, well-drained soil with large particles         |

### How It Works

1. Frontend sends a **base64-encoded image** via POST
2. Image is decoded, converted to RGB, and **resized to 256×256**
3. Pixel values are **normalized to [0, 1]**
4. The CNN model predicts a soil class
5. The predicted soil type is matched against `data_core.csv` to fetch **3–5 recommended crops**
6. Response is returned as JSON

---

## 🚀 Getting Started

### Prerequisites

- [Python](https://www.python.org/) ≥ 3.10
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux / macOS
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
python main.py
```

The server starts at `http://localhost:5000`.

### Running with Gunicorn (Production)

```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

---

## 📡 API Reference

### Health Check

```
GET /
```

**Response:**

```json
{ "status": "Soil Classifier API is running!" }
```

---

### Classify Soil

```
POST /api/classify
Content-Type: application/json
```

**Request Body:**

```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/4AAQ...",
  "language": "English"
}
```

| Field         | Type     | Required | Description                                  |
| ------------- | -------- | -------- | -------------------------------------------- |
| `imageBase64` | `string` | ✅       | Base64-encoded image (with or without data URI prefix) |
| `language`    | `string` | ❌       | Response language (`English`, `Hindi`, `Odia`). Defaults to `English` |

**Success Response (200):**

```json
{
  "soilType": "Loamy",
  "explanation": "Based on the visual characteristics evaluated by the CNN model, this appears to be Loamy soil.",
  "recommendedCrops": ["Wheat", "Rice", "Sugarcane", "Maize"]
}
```

**Error Response (400):**

```json
{ "error": "No image provided" }
```

**Error Response (500):**

```json
{ "error": "Error message details" }
```

---

## 🐳 Docker

### Build

```bash
docker build -t soil-classifier-backend .
```

### Run

```bash
docker run -p 7860:7860 soil-classifier-backend
```

The API will be available at `http://localhost:7860`.

> The Dockerfile is pre-configured for **Hugging Face Spaces** (port 7860). Change the `PORT` environment variable for other platforms.

---

## ☁️ Deployment

### Hugging Face Spaces

1. Create a new Space with **Docker** SDK
2. Push this backend directory to the Space repo
3. The `Dockerfile` handles everything automatically

### Railway

The `railway.toml` is pre-configured:

```toml
[build]
builder = "NIXPACKS"
buildCommand = "pip install -r requirements.txt"

[deploy]
startCommand = "gunicorn --bind 0.0.0.0:$PORT main:app"
restartPolicyType = "ON_FAILURE"
```

Simply connect the repo to Railway and deploy.

### Heroku

The `Procfile` is included:

```
web: gunicorn main:app
```

```bash
heroku create
git push heroku main
```

---

## 📊 Dataset

The `data/data_core.csv` file contains **soil-to-crop mappings** with the following structure:

| Column      | Description                        |
| ----------- | ---------------------------------- |
| `Soil Type` | One of the 5 supported soil types  |
| `Crop Type` | A crop suitable for that soil type |

The API randomly selects **3–5 unique crops** from the matching soil type rows.

---

## ⚙️ Environment Variables

| Variable | Default | Description                        |
| -------- | ------- | ---------------------------------- |
| `PORT`   | `5000`  | Port the server listens on         |

---

## 🧑‍🤝‍🧑 Team

**Team 12 — NIELIT Internship Project**

---

## 📄 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
