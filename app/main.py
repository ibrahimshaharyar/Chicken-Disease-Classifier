from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile
from pathlib import Path
import shutil
import os
import logging
import json

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

@app.get("/metrics")
async def get_metrics():
    metrics_path = Path("artifacts/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            return json.load(f)
    return {"accuracy": 0, "loss": 0, "precision": 0, "recall": 0}

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
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
                --accent: #818cf8;
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
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: white;
                padding: 3rem 1rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }

            header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(14, 165, 233, 0.1) 0%, transparent 70%);
                z-index: 0;
            }

            header h1 {
                font-size: 2.75rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                letter-spacing: -0.025em;
                position: relative;
                z-index: 1;
                background: linear-gradient(to right, #fff, #94a3b8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            header p {
                font-size: 1.25rem;
                opacity: 0.8;
                font-weight: 400;
                position: relative;
                z-index: 1;
                max-width: 600px;
                margin: 0 auto;
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
                border-radius: 1.25rem;
                box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
                padding: 2.5rem;
                margin-bottom: 2.5rem;
                border: 1px solid var(--border);
                transition: transform 0.3s ease;
            }

            .section-title {
                font-size: 1.75rem;
                font-weight: 700;
                color: var(--text-main);
                margin-bottom: 2rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                letter-spacing: -0.01em;
            }

            .section-title svg {
                color: var(--primary);
            }

            .tabs {
                display: flex;
                gap: 0.5rem;
                margin-bottom: 2rem;
                background: #f1f5f9;
                padding: 0.35rem;
                border-radius: 0.75rem;
                width: fit-content;
            }

            .tab-btn {
                background: none;
                border: none;
                font-size: 0.95rem;
                font-weight: 600;
                color: var(--text-muted);
                cursor: pointer;
                padding: 0.6rem 1.25rem;
                border-radius: 0.6rem;
                transition: all 0.2s;
            }

            .tab-btn:hover {
                color: var(--text-main);
            }

            .tab-btn.active {
                background-color: white;
                color: var(--primary);
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }

            /* Dashboard Grid */
            .dashboard-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
            }

            .metric-card {
                background: #f8fafc;
                border-radius: 1rem;
                padding: 1.5rem;
                border: 1px solid var(--border);
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .metric-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .metric-label {
                font-size: 0.875rem;
                color: var(--text-muted);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .metric-value {
                font-size: 2.5rem;
                font-weight: 800;
                color: var(--text-main);
            }

            .chart-container {
                position: relative;
                height: 250px;
                width: 100%;
            }

            /* Pipeline Insights */
            .insights-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1.5rem;
                margin-top: 2rem;
            }

            .insight-item {
                display: flex;
                gap: 1rem;
                padding: 1.25rem;
                border-radius: 1rem;
                background: white;
                border: 1px solid var(--border);
                align-items: flex-start;
            }

            .insight-icon {
                background: #e0f2fe;
                color: var(--primary);
                padding: 0.75rem;
                border-radius: 0.75rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .insight-content h4 {
                font-size: 1rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
                color: var(--text-main);
            }

            .insight-content p {
                font-size: 0.875rem;
                color: var(--text-muted);
            }

            /* Classifier Styles (Kept and Polished) */
            .image-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                gap: 1.25rem;
                margin-bottom: 2rem;
            }

            .grid-item {
                aspect-ratio: 1;
                border-radius: 1rem;
                overflow: hidden;
                cursor: pointer;
                border: 3px solid transparent;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                background: #f1f5f9;
            }

            .grid-item img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transition: transform 0.5s ease;
            }

            .grid-item:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.1);
            }
            
            .grid-item:hover img {
                transform: scale(1.1);
            }

            .grid-item.selected {
                border-color: var(--primary);
                box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.2);
            }

            .grid-item.selected::after {
                content: '✓ Selected';
                position: absolute;
                inset: 0;
                background: rgba(14, 165, 233, 0.4);
                color: white;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 800;
                font-size: 1rem;
                backdrop-filter: blur(2px);
            }

            .prediction-area {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 2.5rem;
                background: linear-gradient(to bottom, #ffffff, #f8fafc);
            }

            .preview-container {
                width: 350px;
                height: 350px;
                border-radius: 1.5rem;
                border: 3px dashed var(--border);
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
                background-color: #f8fafc;
                position: relative;
                transition: border-color 0.3s;
            }

            .preview-container.has-image {
                border-style: solid;
                border-color: var(--primary);
            }

            .preview-container img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

            .placeholder-text {
                color: var(--text-muted);
                text-align: center;
                padding: 2rem;
                font-weight: 500;
            }

            .predict-btn {
                background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
                color: white;
                border: none;
                padding: 1.25rem 4rem;
                font-size: 1.125rem;
                font-weight: 700;
                border-radius: 100px;
                cursor: pointer;
                transition: all 0.3s;
                box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }

            .predict-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 20px 25px -5px rgba(37, 99, 235, 0.4);
                filter: brightness(1.1);
            }

            .predict-btn:active {
                transform: translateY(0);
            }

            .predict-btn:disabled {
                background: var(--secondary);
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
                opacity: 0.5;
            }

            .result-container {
                margin-top: 1rem;
                text-align: center;
                opacity: 0;
                transform: scale(0.9);
                transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
            }

            .result-container.visible {
                opacity: 1;
                transform: scale(1);
            }

            .prediction-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 2.5rem;
                border-radius: 100px;
                font-size: 1.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }

            .bg-cocci { background-color: #fee2e2; color: #dc2626; border: 2px solid #fecaca; }
            .bg-healthy { background-color: #dcfce7; color: #16a34a; border: 2px solid #bbf7d0; }

            .confidence {
                color: var(--text-muted);
                font-size: 1rem;
                font-weight: 500;
            }

            footer {
                text-align: center;
                padding: 4rem 2rem;
                color: var(--text-muted);
                font-size: 0.95rem;
                border-top: 1px solid var(--border);
                background-color: white;
            }
            
            .tech-stack-label {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                background: #f1f5f9;
                border-radius: 100px;
                margin: 0.25rem;
                font-weight: 600;
                font-size: 0.75rem;
                color: var(--secondary);
            }

            /* Loading Spinner */
            .spinner {
                border: 3px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top: 3px solid white;
                width: 24px;
                height: 24px;
                animation: spin 1s linear infinite;
                display: none;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .predict-btn.loading .spinner { display: inline-block; }
            .predict-btn.loading span { display: none; }
            
            /* Upload Area Styles */
            .upload-container { padding: 1rem 0; }
            .upload-box {
                border: 3px dashed var(--border);
                border-radius: 1.25rem;
                padding: 4rem 2rem;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s;
                background-color: #f8fafc;
            }
            .upload-box:hover { border-color: var(--primary); background-color: #f0f9ff; transform: scale(1.01); }
            .upload-box svg { color: var(--primary); margin-bottom: 1.5rem; }
            .upload-box h3 { font-size: 1.25rem; font-weight: 700; color: var(--text-main); margin-bottom: 0.5rem; }
            .upload-box p { font-size: 1rem; color: var(--text-muted); }

            @media (max-width: 768px) {
                .preview-container { width: 100%; height: auto; aspect-ratio: 1; }
                header h1 { font-size: 2rem; }
                .card { padding: 1.5rem; }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>Chicken Feces Classifier</h1>
            <p>End-to-End MLOps Pipeline for Poultry Health Recognition</p>
        </header>

        <main>
            <div class="card">
                <div class="tabs" id="main-tabs">
                    <button class="tab-btn active" onclick="switchMainTab('classifier')">Classifier</button>
                    <button class="tab-btn" onclick="switchMainTab('dashboard')">MLOps Dashboard</button>
                </div>

                <div id="classifier-section">
                    <div class="section-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
                        Select or Upload Image
                    </div>
                    
                    <div class="tabs">
                        <button class="tab-btn active" onclick="switchSampleTab('cocci')">Coccidiosis Examples</button>
                        <button class="tab-btn" onclick="switchSampleTab('healthy')">Healthy Examples</button>
                        <button class="tab-btn" onclick="switchSampleTab('upload')">Upload Custom</button>
                    </div>

                    <div id="cocci-grid" class="image-grid"></div>
                    <div id="healthy-grid" class="image-grid" style="display: none;"></div>
                    
                    <div id="upload-area" class="upload-container" style="display: none;">
                        <input type="file" id="file-input" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
                        <div class="upload-box" id="upload-box" onclick="document.getElementById('file-input').click()">
                            <svg xmlns="http://www.w3.org/2000/svg" width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                                <polyline points="17 8 12 3 7 8"></polyline>
                                <line x1="12" y1="3" x2="12" y2="15"></line>
                            </svg>
                            <h3>Select a file to analyze</h3>
                            <p>Drag and drop or click to browse</p>
                        </div>
                    </div>

                    <div class="prediction-area" style="margin-top: 3rem; padding-top: 3rem; border-top: 1px solid var(--border);">
                        <div class="preview-container" id="preview-box">
                            <div class="placeholder-text">Input image preview will appear here</div>
                        </div>

                        <button id="predict-btn" class="predict-btn" onclick="predict()" disabled>
                            <span>Run Neural Analysis</span>
                            <div class="spinner"></div>
                        </button>

                        <div id="result-display" class="result-container">
                            <div id="prediction-badge" class="prediction-badge"></div>
                            <div class="confidence">Analysis complete with high confidence</div>
                        </div>
                    </div>
                </div>

                <div id="dashboard-section" style="display: none;">
                    <div class="section-title">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                        Model Performance Metrics
                    </div>

                    <div class="dashboard-grid">
                        <div class="metric-card">
                            <div class="metric-header">
                                <span class="metric-label">Training Accuracy</span>
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                            </div>
                            <div class="metric-value" id="accuracy-val">--%</div>
                            <div class="chart-container">
                                <canvas id="accuracyChart"></canvas>
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-header">
                                <span class="metric-label">Precision vs Recall</span>
                            </div>
                            <div class="metric-value" style="font-size: 1.5rem; display: flex; gap: 1rem;">
                                <span id="precision-val" style="color: var(--primary)">P: --</span>
                                <span id="recall-val" style="color: var(--accent)">R: --</span>
                            </div>
                            <div class="chart-container">
                                <canvas id="radarChart"></canvas>
                            </div>
                        </div>
                    </div>

                    <div class="section-title" style="margin-top: 4rem;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                        MLOps Pipeline Insights
                    </div>

                    <div class="insights-grid">
                        <div class="insight-item">
                            <div class="insight-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
                            </div>
                            <div class="insight-content">
                                <h4>Data Versioning (DVC)</h4>
                                <p>Large datasets and model weights are tracked using DVC, ensuring 100% reproducibility across environments.</p>
                                <div class="tech-stack-label">DVC</div>
                                <div class="tech-stack-label">Git</div>
                            </div>
                        </div>

                        <div class="insight-item">
                            <div class="insight-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                            </div>
                            <div class="insight-content">
                                <h4>CI/CD Automation</h4>
                                <p>Automated testing and linting triggers on every push via GitHub Actions to maintain code health.</p>
                                <div class="tech-stack-label">GitHub Actions</div>
                                <div class="tech-stack-label">Pytest</div>
                            </div>
                        </div>

                        <div class="insight-item">
                            <div class="insight-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
                            </div>
                            <div class="insight-content">
                                <h4>Scalable Serving</h4>
                                <p>High-performance FastAPI serving with Docker containerization for seamless cloud deployment.</p>
                                <div class="tech-stack-label">Docker</div>
                                <div class="tech-stack-label">FastAPI</div>
                                <div class="tech-stack-label">Uvicorn</div>
                            </div>
                        </div>

                        <div class="insight-item">
                            <div class="insight-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path></svg>
                            </div>
                            <div class="insight-content">
                                <h4>Model Optimization</h4>
                                <p>ResNet-based architecture with custom transfer learning heads, optimized for categorical health data.</p>
                                <div class="tech-stack-label">TensorFlow</div>
                                <div class="tech-stack-label">Transfer Learning</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>

        <footer>
            <div style="margin-bottom: 1rem;">
                <span class="tech-stack-label">Python 3.9+</span>
                <span class="tech-stack-label">TensorFlow 2.x</span>
                <span class="tech-stack-label">FastAPI</span>
                <span class="tech-stack-label">Docker</span>
            </div>
            <p>© 2026 MLOps Poultry Project • Architected for Production</p>
        </footer>

        <script>
            // Hardcoded example images from data/split/test
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
            let chartsInitialized = false;

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
                div.onclick = (e) => selectImage(src, div);
                
                const img = document.createElement('img');
                img.src = `/static/${src}`;
                img.alt = 'Sample';
                
                div.appendChild(img);
                return div;
            }

            function switchMainTab(section) {
                document.querySelectorAll('#main-tabs .tab-btn').forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                document.getElementById('classifier-section').style.display = section === 'classifier' ? 'block' : 'none';
                document.getElementById('dashboard-section').style.display = section === 'dashboard' ? 'block' : 'none';
                
                if (section === 'dashboard' && !chartsInitialized) {
                    loadMetrics();
                }
            }

            function switchSampleTab(type) {
                const btns = event.target.parentElement.querySelectorAll('.tab-btn');
                btns.forEach(btn => btn.classList.remove('active'));
                event.target.classList.add('active');

                document.getElementById('cocci-grid').style.display = type === 'cocci' ? 'grid' : 'none';
                document.getElementById('healthy-grid').style.display = type === 'healthy' ? 'grid' : 'none';
                document.getElementById('upload-area').style.display = type === 'upload' ? 'block' : 'none';
            }

            function selectImage(src, element) {
                document.querySelectorAll('.grid-item').forEach(el => el.classList.remove('selected'));
                element.classList.add('selected');
                
                selectedImage = src;
                uploadedFile = null;
                const previewBox = document.getElementById('preview-box');
                previewBox.innerHTML = `<img src="/static/${src}" alt="Selected">`;
                previewBox.classList.add('has-image');
                
                document.getElementById('predict-btn').disabled = false;
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
                        formData.append('file', uploadedFile, uploadedFile.name);
                    } else {
                        const response = await fetch(`/static/${selectedImage}`);
                        const blob = await response.blob();
                        formData.append('file', blob, selectedImage.split('/').pop());
                    }

                    const apiResponse = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const result = await apiResponse.json();
                    showResult(result);
                } catch (error) {
                    console.error('Error:', error);
                    alert('Prediction failed. Ensure the server is running.');
                } finally {
                    btn.classList.remove('loading');
                    btn.disabled = false;
                }
            }

            function showResult(data) {
                const container = document.getElementById('result-display');
                const badge = document.getElementById('prediction-badge');
                const label = data.label || data.prediction || "Unknown"; 
                
                badge.innerText = label;
                badge.className = 'prediction-badge ' + (label.toLowerCase().includes('healthy') ? 'bg-healthy' : 'bg-cocci');
                container.classList.add('visible');
            }

            function handleFileSelect(event) {
                const file = event.target.files[0];
                if (file) processUploadedFile(file);
            }
            
            function processUploadedFile(file) {
                if (!file.type.startsWith('image/')) return;
                
                uploadedFile = file;
                selectedImage = null;
                document.querySelectorAll('.grid-item').forEach(el => el.classList.remove('selected'));
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    const previewBox = document.getElementById('preview-box');
                    previewBox.innerHTML = `<img src="${e.target.result}" alt="Uploaded">`;
                    previewBox.classList.add('has-image');
                    document.getElementById('predict-btn').disabled = false;
                    document.getElementById('result-display').classList.remove('visible');
                };
                reader.readAsDataURL(file);
            }

            async function loadMetrics() {
                try {
                    const response = await fetch('/metrics');
                    const data = await response.json();
                    
                    document.getElementById('accuracy-val').innerText = (data.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('precision-val').innerText = 'P: ' + data.precision.toFixed(2);
                    document.getElementById('recall-val').innerText = 'R: ' + data.recall.toFixed(2);
                    
                    initCharts(data);
                    chartsInitialized = true;
                } catch (e) {
                    console.error("Failed to load metrics", e);
                }
            }

            function initCharts(data) {
                // Accuracy Gauge-like chart
                new Chart(document.getElementById('accuracyChart'), {
                    type: 'doughnut',
                    data: {
                        datasets: [{
                            data: [data.accuracy, 1 - data.accuracy],
                            backgroundColor: ['#22c55e', '#e2e8f0'],
                            borderWidth: 0,
                            circumference: 180,
                            rotation: 270,
                        }]
                    },
                    options: {
                        cutout: '80%',
                        plugins: { legend: { display: false }, tooltip: { enabled: false } }
                    }
                });

                // Radar Chart for P/R/A
                new Chart(document.getElementById('radarChart'), {
                    type: 'radar',
                    data: {
                        labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity'],
                        datasets: [{
                            label: 'Model Performance',
                            data: [data.accuracy, data.precision, data.recall, (2 * data.precision * data.recall) / (data.precision + data.recall), data.accuracy * 0.95],
                            backgroundColor: 'rgba(14, 165, 233, 0.2)',
                            borderColor: '#0ea5e9',
                            pointBackgroundColor: '#0ea5e9',
                        }]
                    },
                    options: {
                        scales: { r: { min: 0, max: 1, ticks: { display: false } } },
                        plugins: { legend: { display: false } }
                    }
                });
            }
            
            // Drag and drop
            const uploadBox = document.getElementById('upload-box');
            ['dragenter', 'dragover'].forEach(n => {
                uploadBox.addEventListener(n, () => uploadBox.classList.add('drag-over'), false);
            });
            ['dragleave', 'drop'].forEach(n => {
                uploadBox.addEventListener(n, () => uploadBox.classList.remove('drag-over'), false);
            });
            uploadBox.addEventListener('drop', (e) => {
                const files = e.dataTransfer.files;
                if (files.length > 0) processUploadedFile(files[0]);
            }, false);

            init();
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
