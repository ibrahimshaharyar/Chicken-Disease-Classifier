# Chicken Feces Coccidiosis Classification

This project provides an end-to-end MLOps pipeline for classifying Chicken Feces images into **Healthy** or **Coccidiosis** categories using Deep Learning.

## Features
- **Data Ingestion**: Automated downloading and extraction of dataset.
- **Data Splitting**: Systematic splitting into Train, Validation, and Test sets.
- **Model Training**: TensorFlow/Keras based training with checkpointing and early stopping.
- **API Serving**: FastAPI application for real-time inference.
- **Docker Support**: Containerized application for easy deployment.
- **CI/CD**: GitHub Actions for automated testing and deployment.

## Project Structure
```
├── app/                # FastAPI application
├── src/                # Source code
│   ├── data/           # Data processing scripts (ingest, split)
│   ├── model/          # Model definition and training
│   ├── serving/        # Inference logic
├── configs/            # Configuration files (params.yaml)
├── artifacts/          # Training artifacts (models, metrics)
├── tests/              # Unit tests
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites
- Python 3.9+
- TensorFlow
- Docker (optional)

### Installation
1. Clone the repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Pipeline

### 1. Data Ingestion
Download and extract the dataset:
```bash
python src/data/ingest.py
```

### 2. Data Splitting
Split data into Train, Validation, and Test sets:
```bash
python src/data/split.py
```

### 3. Training
Train the model:
```bash
python src/model/train.py
```
The model will be saved in `artifacts/model/`.

### 4. API Serving
Run the API locally:
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000` for the UI or `http://localhost:8000/docs` for the API documentation.

## Docker Deployment

Build the image:
```bash
docker build -t chicken-classifier .
```

Run the container:
```bash
docker run -p 8000:8000 chicken-classifier
```

## CI/CD
This project uses GitHub Actions for CI/CD. The workflows are defined in `.github/workflows/`.
