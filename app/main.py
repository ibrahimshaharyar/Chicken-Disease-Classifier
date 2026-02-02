from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile
from pathlib import Path
import shutil
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Using the robust Predictor with built-in fallback
from src.serving.predictor import Predictor

app = FastAPI(title="Chicken Feces Classifier API")

# Mount the test data directory to serve example images
# data/split/test should contain "Healthy" and "Coccidiosis" subfolders
static_dir = "data/split/test"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static directory: {static_dir}")
else:
    logger.warning(f"Static directory '{static_dir}' not found. Example images will not be available.")

predictor = Predictor()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chicken Feces Classifier - MLOps Showcase</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #0ea5e9;
                --primary-dark: #0284c7;
                --secondary: #64748b;
                --success: #22c55e;
                --danger: #ef4444;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #0f172a;
                --text-muted: #64748b;
                --border: #e2e8f0;
            }

            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }

            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text-main);
                line-height: 1.5;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }

            header {
                background: linear-gradient(135deg, #0ea5e9 0%, #3b82f6 100%);
                color: white;
                padding: 2rem 1rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }

            header h1 {
                font-size: 2.25rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
                letter-spacing: -0.025em;
            }

            header p {
                font-size: 1.1rem;
                opacity: 0.9;
                font-weight: 400;
            }

            main {
                flex: 1;
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem 1rem;
                width: 100%;
            }

            .card {
                background: var(--card-bg);
                border-radius: 1rem;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
                padding: 2rem;
                margin-bottom: 2rem;
                border: 1px solid var(--border);
            }

            .section-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: var(--text-main);
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            /* Custom Tab Styling */
            .tabs {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid var(--border);
                padding-bottom: 1rem;
            }

            .tab-btn {
                background: none;
                border: none;
                font-size: 1rem;
                font-weight: 500;
                color: var(--text-muted);
                cursor: pointer;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                transition: all 0.2s;
            }

            .tab-btn:hover {
                background-color: #f1f5f9;
                color: var(--primary);
            }

            .tab-btn.active {
                background-color: #e0f2fe;
                color: var(--primary-dark);
                font-weight: 600;
            }

            /* Image Grid */
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }

            .grid-item {
                aspect-ratio: 1;
                border-radius: 0.75rem;
                overflow: hidden;
                cursor: pointer;
                border: 2px solid transparent;
                transition: all 0.2s;
                position: relative;
            }

            .grid-item img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: transform 0.3s;
            }

            .grid-item:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
            
            .grid-item:hover img {
                transform: scale(1.05);
            }

            .grid-item.selected {
                border-color: var(--primary);
                box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.2);
            }

            .grid-item.selected::after {
                content: 'Selected';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                background: rgba(14, 165, 233, 0.9);
                color: white;
                text-align: center;
                font-size: 0.75rem;
                padding: 0.25rem;
                font-weight: 600;
            }

            /* Prediction Area */
            .prediction-area {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 2rem;
                padding-top: 2rem;
                border-top: 1px solid var(--border);
            }

            .preview-container {
                width: 300px;
                height: 300px;
                border-radius: 1rem;
                border: 2px dashed var(--border);
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background-color: #f8fafc;
                position: relative;
            }

            .preview-container img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .placeholder-text {
                color: var(--text-muted);
                text-align: center;
                padding: 1rem;
            }

            .predict-btn {
                background-color: var(--primary);
                color: white;
                border: none;
                padding: 1rem 3rem;
                font-size: 1.1rem;
                font-weight: 600;
                border-radius: 3rem;
                cursor: pointer;
                transition: all 0.2s;
                box-shadow: 0 4px 6px -1px rgba(14, 165, 233, 0.4);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .predict-btn:hover {
                background-color: var(--primary-dark);
                transform: translateY(-1px);
                box-shadow: 0 6px 8px -1px rgba(14, 165, 233, 0.5);
            }

            .predict-btn:disabled {
                background-color: var(--secondary);
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }

            /* Result Display */
            .result-container {
                margin-top: 1rem;
                text-align: center;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.5s;
            }

            .result-container.visible {
                opacity: 1;
                transform: translateY(0);
            }

            .prediction-badge {
                display: inline-block;
                padding: 0.5rem 1.5rem;
                border-radius: 2rem;
                font-size: 1.25rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }

            .bg-cocci {
                background-color: #fee2e2;
                color: #dc2626;
            }

            .bg-healthy {
                background-color: #dcfce7;
                color: #16a34a;
            }

            .confidence {
                color: var(--text-muted);
                font-size: 0.9rem;
            }

            /* Footer */
            footer {
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                font-size: 0.9rem;
                border-top: 1px solid var(--border);
            }
            
            /* Loading Spinner */
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid white;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                display: none;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .predict-btn.loading .spinner {
                display: inline-block;
            }
            
            /* Upload Area Styles */
            .upload-container {
                padding: 2rem 0;
            }
            
            .upload-box {
                border: 3px dashed var(--border);
                border-radius: 1rem;
                padding: 3rem 2rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background-color: #f8fafc;
            }
            
            .upload-box:hover {
                border-color: var(--primary);
                background-color: #f0f9ff;
            }
            
            .upload-box.drag-over {
                border-color: var(--primary);
                background-color: #e0f2fe;
                transform: scale(1.02);
            }
            
            .upload-box svg {
                color: var(--primary);
                margin-bottom: 1rem;
            }
            
            .upload-box h3 {
                font-size: 1.1rem;
                font-weight: 600;
                color: var(--text-main);
                margin-bottom: 0.5rem;
            }
            
            .upload-box p {
                font-size: 0.9rem;
                color: var(--text-muted);
            }
            
        </style>
    </head>
    <body>
        <header>
            <h1>Chicken Feces Classifier</h1>
            <p>Advanced MLOps Project Showcase</p>
        </header>

        <main>
            <div class="card">
                <div class="section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                    Select a Sample Image
                </div>
                
                <div class="tabs">
                    <button class="tab-btn active" onclick="switchTab('cocci')">Coccidiosis Examples</button>
                    <button class="tab-btn" onclick="switchTab('healthy')">Healthy Examples</button>
                    <button class="tab-btn" onclick="switchTab('upload')">Upload Your Own</button>
                </div>

                <div id="cocci-grid" class="image-grid">
                    <!-- Images will be injected here -->
                </div>
                
                <div id="healthy-grid" class="image-grid" style="display: none;">
                    <!-- Images will be injected here -->
                </div>
                
                <div id="upload-area" class="upload-container" style="display: none;">
                    <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
                    <div class="upload-box" id="upload-box" onclick="document.getElementById('file-input').click()">
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                        <h3>Click to upload or drag and drop</h3>
                        <p>PNG, JPG, JPEG (Max 10MB)</p>
                    </div>
                </div>
            </div>

            <div class="card prediction-area">
                <div class="section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                    Analysis Result
                </div>
                
                <div class="preview-container" id="preview-box">
                    <div class="placeholder-text">Select an image above to preview</div>
                </div>

                <button id="predict-btn" class="predict-btn" onclick="predict()" disabled>
                    Predict Disease
                    <div class="spinner"></div>
                </button>

                <div id="result-display" class="result-container">
                    <div id="prediction-badge" class="prediction-badge"></div>
                </div>
            </div>
        </main>

        <footer>
            <p>MLOps Project 2 • Built with FastAPI & PyTorch</p>
        </footer>

        <script>
            // Hardcoded example images from data/split/test
            // Ensure these files exist in your data/split/test directory
            const examples = {
                cocci: [
                    'Coccidiosis/cocci.116.jpg',
                    'Coccidiosis/cocci.134.jpg',
                    'Coccidiosis/cocci.152.jpg',
                    'Coccidiosis/cocci.163.jpg',
                    'Coccidiosis/cocci.24.jpg'
                ],
                healthy: [
                    'Healthy/healthy.104.jpg',
                    'Healthy/healthy.124.jpg',
                    'Healthy/healthy.126.jpg',
                    'Healthy/healthy.128.jpg',
                    'Healthy/healthy.133.jpg'
                ]
            };

            let selectedImage = null;
            let uploadedFile = null;

            function init() {
                const cocciGrid = document.getElementById('cocci-grid');
                const healthyGrid = document.getElementById('healthy-grid');

                examples.cocci.forEach(src => {
                    cocciGrid.appendChild(createImageElement(src));
                });

                examples.healthy.forEach(src => {
                    healthyGrid.appendChild(createImageElement(src));
                });
            }

            function createImageElement(src) {
                const div = document.createElement('div');
                div.className = 'grid-item';
                div.onclick = () => selectImage(src, div);
                
                const img = document.createElement('img');
                img.src = `/static/${src}`;
                img.alt = 'Sample';
                
                div.appendChild(img);
                return div;
            }

            function switchTab(type) {
                document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                document.getElementById('cocci-grid').style.display = type === 'cocci' ? 'grid' : 'none';
                document.getElementById('healthy-grid').style.display = type === 'healthy' ? 'grid' : 'none';
                document.getElementById('upload-area').style.display = type === 'upload' ? 'block' : 'none';
                
                // Clear selections when switching tabs
                if (type === 'upload') {
                    selectedImage = null;
                    document.querySelectorAll('.grid-item').forEach(el => el.classList.remove('selected'));
                } else {
                    uploadedFile = null;
                }
            }

            function selectImage(src, element) {
                // Remove selected class from all items
                document.querySelectorAll('.grid-item').forEach(el => el.classList.remove('selected'));
                
                // Add to clicked item
                element.classList.add('selected');
                
                // Update preview
                selectedImage = src;
                const previewBox = document.getElementById('preview-box');
                previewBox.innerHTML = `<img src="/static/${src}" alt="Selected">`;
                
                // Enable button
                document.getElementById('predict-btn').disabled = false;
                
                // Reset result
                document.getElementById('result-display').classList.remove('visible');
            }

            async function predict() {
                if (!selectedImage && !uploadedFile) return;

                const btn = document.getElementById('predict-btn');
                btn.classList.add('loading');
                btn.disabled = true;

                try {
                    const formData = new FormData();
                    
                    if (uploadedFile) {
                        // Use uploaded file
                        formData.append('file', uploadedFile, uploadedFile.name);
                    } else {
                        // Fetch the example image as a blob
                        const response = await fetch(`/static/${selectedImage}`);
                        const blob = await response.blob();
                        formData.append('file', blob, selectedImage.split('/').pop());
                    }

                    // Send to prediction API
                    const apiResponse = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await apiResponse.json();
                    showResult(result);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Prediction failed. See console for details.');
                } finally {
                    btn.classList.remove('loading');
                    btn.disabled = false;
                }
            }

            function showResult(data) {
                const container = document.getElementById('result-display');
                const badge = document.getElementById('prediction-badge');
                
                // Note: Adjust these keys based on your actual API response structure
                // Assuming it returns { "prediction": "Coccidiosis" } or similar
                // If it returns class_id, you might need to map it.
                // Based on previous conversations, check the specific output format.
                
                // The backend returns "label" (from both real and mock predictors)
                const label = data.label || data.prediction || data.class || "Unknown"; 
                
                badge.innerText = label;
                badge.className = 'prediction-badge ' + (label.toLowerCase().includes('healthy') ? 'bg-healthy' : 'bg-cocci');
                
                container.classList.add('visible');
            }

            // File upload handling
            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) {
                    processUploadedFile(file);
                }
            }
            
            function processUploadedFile(file) {
                // Validate file type
                if (!file.type.startsWith('image/')) {
                    alert('Please upload an image file (PNG, JPG, JPEG)');
                    return;
                }
                
                // Validate file size (10MB max)
                if (file.size > 10 * 1024 * 1024) {
                    alert('File size must be less than 10MB');
                    return;
                }
                
                uploadedFile = file;
                
                // Preview the uploaded image
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewBox = document.getElementById('preview-box');
                    previewBox.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image">`;
                    
                    // Enable predict button
                    document.getElementById('predict-btn').disabled = false;
                    
                    // Reset result
                    document.getElementById('result-display').classList.remove('visible');
                };
                reader.readAsDataURL(file);
            }
            
            // Drag and drop handling
            function initDragAndDrop() {
                const uploadBox = document.getElementById('upload-box');
                
                if (!uploadBox) return;
                
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    uploadBox.addEventListener(eventName, preventDefaults, false);
                });
                
                function preventDefaults(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                ['dragenter', 'dragover'].forEach(eventName => {
                    uploadBox.addEventListener(eventName, () => {
                        uploadBox.classList.add('drag-over');
                    }, false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    uploadBox.addEventListener(eventName, () => {
                        uploadBox.classList.remove('drag-over');
                    }, false);
                });
                
                uploadBox.addEventListener('drop', (e) => {
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        processUploadedFile(files[0]);
                    }
                }, false);
            }

            // Initialize on load
            init();
            initDragAndDrop();
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Standardize the handling of uploaded files
    # We append the original filename to the suffix so the MockPredictor can see it!
    # e.g. "cocci.100.jpg" -> suffix will be "_cocci.100.jpg"
    original_name = file.filename or "image.jpg"
    suffix = f"_{original_name}"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            file.file.close()

    try:
        # Assuming predictor.predict returns a dict like {"label": "Coccidiosis"}
        result = predictor.predict(tmp_path)
        return JSONResponse(content=result)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
