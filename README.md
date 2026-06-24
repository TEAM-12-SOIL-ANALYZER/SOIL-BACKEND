# 🌱 Soil Classification Backend

> **AI-powered soil classification API** — Upload a soil image, get the soil type and crop recommendations in seconds.

A production-ready **FastAPI** backend that uses a custom-trained **CNN (Convolutional Neural Network)** to classify soil images into five categories and recommend suitable crops based on a curated agricultural dataset. Supports multilingual responses in **English**, **Hindi**, and **Odia**.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **CNN Classification** | Classifies soil into 5 types — Black, Clayey, Loamy, Red, Sandy |
| 🌾 **Crop Recommendations** | Returns suitable crops from 8 000+ soil-to-crop mappings |
| 🌐 **Multilingual** | Explanations in English, Hindi, and Odia |
| ⚡ **Fast Inference** | Single-request latency under 500 ms on CPU |
| 🐳 **Docker Ready** | One-command deployment via Docker |
| 🤗 **Hugging Face Spaces** | Deployed as a Docker-based HF Space |

---

## 🏗️ Architecture

```
┌─────────────┐    base64 image     ┌────────────────────────┐
│   Frontend   │ ─────────────────▶ │  FastAPI  /api/classify │
│  (any client)│ ◀───────────────── │                        │
└─────────────┘   JSON response     │  ┌──────────────────┐  │
                                    │  │  CNN Model        │  │
                                    │  │  (TensorFlow/     │  │
                                    │  │   Keras)          │  │
                                    │  └──────────────────┘  │
                                    │  ┌──────────────────┐  │
                                    │  │  Crop CSV Lookup  │  │
                                    │  └──────────────────┘  │
                                    └────────────────────────┘
```

---

## 📡 API Reference

### Health Check

```
GET /
```

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "soil_classes": ["Black", "Clayey", "Loamy", "Red", "Sandy"]
}
```

---

### Classify Soil

```
POST /api/classify
```

**Request Body**

```json
{
  "imageBase64": "data:image/jpeg;base64,/9j/4AAQ...",
  "language": "English"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `imageBase64` | `string` | *required* | Base64-encoded soil image (data URI or raw base64) |
| `language` | `string` | `"English"` | Response language — `English`, `Hindi`, or `Odia` |

**Response**

```json
{
  "soilType": "Sandy",
  "explanation": "This soil appears to be Sandy soil. It has large, loose particles...",
  "recommendedCrops": ["Barley", "Maize", "Millets"]
}
```

| Field | Type | Description |
|---|---|---|
| `soilType` | `string` | Predicted soil class |
| `explanation` | `string` | Human-readable explanation in the requested language |
| `recommendedCrops` | `string[]` | Alphabetically sorted list of recommended crops |

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | CNN (Convolutional Neural Network) |
| Framework | TensorFlow / Keras 2.16 |
| Input Size | 224 × 224 × 3 (RGB) |
| Output | 5-class softmax |
| Classes | Black, Clayey, Loamy, Red, Sandy |
| File | `model/soil_model007.keras` (~19 MB) |

---

## 📂 Project Structure

```
web backend/
├── app.py                 # FastAPI application (routes, model loading, inference)
├── Dockerfile             # Container build for deployment
├── requirements.txt       # Python dependencies
├── README.md              # ← You are here (GitHub)
├── HF_README.md           # Hugging Face Space card
├── .dockerignore
├── data/
│   └── data_core.csv      # Soil type → Crop type mapping (8 000+ rows)
└── model/
    └── soil_model007.keras # Trained CNN weights
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **pip**
- (Optional) **Docker**

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/soil-classification-backend.git
cd soil-classification-backend

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

The API will be available at **http://localhost:7860**.

### Docker

```bash
# Build
docker build -t soil-backend .

# Run
docker run -p 7860:7860 soil-backend
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Web Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| ML Framework | [TensorFlow](https://www.tensorflow.org/) 2.16 (CPU) |
| Image Processing | [Pillow](https://pillow.readthedocs.io/) |
| Data Handling | [pandas](https://pandas.pydata.org/) |
| ASGI Server | [Uvicorn](https://www.uvicorn.org/) |
| Containerisation | [Docker](https://www.docker.com/) |
| Deployment | [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## 🌍 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `model/soil_model007.keras` | Path to the Keras model file |
| `DATA_PATH` | `data/data_core.csv` | Path to the crop mapping CSV |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is part of the **NIELIT Internship Programme**.

---

<p align="center">
  Made with ❤️ for Indian agriculture
</p>
