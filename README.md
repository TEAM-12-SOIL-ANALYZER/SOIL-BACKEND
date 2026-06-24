---
title: Soil Classification Backend
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Soil Classification Backend

A FastAPI backend that uses a CNN model to classify soil types from images and recommend suitable crops.

## API Endpoint

### `POST /api/classify`

**Request:**
```json
{
  "imageBase64": "data:image/jpeg;base64,...",
  "language": "English"
}
```

**Response:**
```json
{
  "soilType": "Sandy",
  "explanation": "This soil appears to be Sandy soil...",
  "recommendedCrops": ["Maize", "Barley", "Millets"]
}
```

## Model

- **Architecture:** CNN (Convolutional Neural Network)
- **Classes:** Sandy, Loamy, Black, Red, Clayey
- **Input:** 224×224 RGB soil image

## Data

- `data/data_core.csv` — soil type to crop type mapping (8000+ records)
