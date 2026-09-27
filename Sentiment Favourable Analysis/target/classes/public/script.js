let impactChart = null;
let trendChart = null;

// Global state to track individual model scores for fusion
let latestScores = {
    lr: 0.0,
    db: 0.0,
    face: 0.0
};

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // Range slider update
    const slider = document.getElementById("temp-slider");
    const sliderVal = document.getElementById("temp-val");
    slider.addEventListener("input", (e) => {
        sliderVal.textContent = parseFloat(e.target.value).toFixed(1);
    });

    // Analyze button click
    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.addEventListener("click", performAnalysis);

    // Setup OCR drag & drop zone
    const dropzone = document.getElementById("ocr-dropzone");
    const fileInput = document.getElementById("image-upload");

    dropzone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleOcrFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleOcrFile(e.dataTransfer.files[0]);
        }
    });

    // Setup camera toggle
    const cameraToggleBtn = document.getElementById("camera-toggle-btn");
    cameraToggleBtn.addEventListener("click", toggleCamera);

    // Initial trends and face models load
    loadTrends();
    loadFaceModels();
});

// Switch between Trend and Word Cloud tabs
function switchTab(tabId) {
    // Toggle active tab buttons
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.remove("active");
    });
    event.currentTarget.classList.add("active");

    // Toggle active content divs
    document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.remove("active");
    });
    document.getElementById(tabId).classList.add("active");
}

// REST call to analyze a review
async function performAnalysis() {
    const text = document.getElementById("review-input").value;
    const temp = parseFloat(document.getElementById("temp-slider").value);
    
    if (!text.trim()) {
        alert("Please enter a review to analyze!");
        return;
    }

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text, temperature: temp })
        });

        if (!response.ok) {
            throw new Error("Failed to analyze sentiment.");
        }

        const data = await response.json();
        
        // Save latest text scores
        latestScores.lr = data.baseline.score;
        latestScores.db = data.transformer.score;

        // Update UI
        updateModelUI("lr", data.baseline);
        updateModelUI("db", data.transformer);
        updateImpactChart(data.contributions);
        updateUnifiedUI();

    } catch (error) {
        console.error("Error analyzing:", error);
    }
}

// Update specific model card metrics and gauges
function updateModelUI(prefix, result) {
    const labelEl = document.getElementById(`${prefix}-label`);
    const scoreEl = document.getElementById(`${prefix}-score`);
    const probEl = document.getElementById(`${prefix}-prob`);
    const posGauge = document.getElementById(`${prefix}-gauge-fill-pos`);
    const negGauge = document.getElementById(`${prefix}-gauge-fill-neg`);

    // Label and colors
    labelEl.textContent = result.label;
    labelEl.style.backgroundColor = result.color;

    // Score format
    const scoreVal = result.score;
    scoreEl.textContent = (scoreVal >= 0 ? "+" : "") + scoreVal.toFixed(3);
    scoreEl.style.color = result.color;

    // Probability percentage
    probEl.textContent = (result.probability * 100).toFixed(1) + "%";

    // Gauge animations
    if (scoreVal >= 0) {
        posGauge.style.width = (scoreVal * 100) / 2 + "%";
        negGauge.style.width = "0%";
    } else {
        negGauge.style.width = (Math.abs(scoreVal) * 100) / 2 + "%";
        posGauge.style.width = "0%";
    }
}

// Render word contribution chart using Chart.js
function updateImpactChart(contributions) {
    // Sort contributions by weight absolute value
    contributions.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    
    // Take top 8 most influential words
    const topContr = contributions.slice(0, 8);
    // Sort alphabetically/re-sort by value for better visual balance
    topContr.sort((a, b) => a.weight - b.weight);

    const labels = topContr.map(c => c.word);
    const dataVals = topContr.map(c => c.weight);
    const bgColors = topContr.map(c => c.weight >= 0 ? "rgba(34, 197, 94, 0.7)" : "rgba(239, 68, 68, 0.7)");
    const borderColors = topContr.map(c => c.weight >= 0 ? "#22c55e" : "#ef4444");

    const ctx = document.getElementById("impact-chart").getContext("2d");

    if (impactChart) {
        impactChart.destroy();
    }

    impactChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: dataVals,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc' }
                }
            }
        }
    });
}

// REST call to load sample corpus stats and positive word list
async function loadTrends() {
    try {
        const response = await fetch("/api/trends");
        if (!response.ok) {
            throw new Error("Failed to load trends.");
        }
        const data = await response.json();
        
        renderTrendChart(data.trends);
        renderWordCloud(data.wordCloud);
    } catch (error) {
        console.error("Error loading trends:", error);
    }
}

// Render sentiment trend graph using Chart.js
function renderTrendChart(trends) {
    const indices = trends.map(t => t.index);
    const scores = trends.map(t => t.score);
    
    // Compute 5-review rolling average
    const rollingAvg = [];
    for (let i = 0; i < scores.length; i++) {
        let sum = 0;
        let count = 0;
        for (let j = Math.max(0, i - 4); j <= i; j++) {
            sum += scores[j];
            count++;
        }
        rollingAvg.push(sum / count);
    }

    const ctx = document.getElementById("trend-chart").getContext("2d");

    if (trendChart) {
        trendChart.destroy();
    }

    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: indices,
            datasets: [
                {
                    label: 'Individual Review',
                    data: scores,
                    borderColor: 'rgba(168, 85, 247, 0.3)',
                    backgroundColor: 'rgba(168, 85, 247, 0.5)',
                    pointRadius: 3,
                    borderWidth: 1,
                    showLine: false
                },
                {
                    label: '5-Review Rolling Avg',
                    data: rollingAvg,
                    borderColor: '#22c55e',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    borderWidth: 2.5,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f8fafc' }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Review Index', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'Favorability Score', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    min: -1.0,
                    max: 1.0
                }
            }
        }
    });
}

// Generate the HTML-based word cloud
function renderWordCloud(wordList) {
    const container = document.getElementById("wordcloud-container");
    container.innerHTML = "";

    // Maximum frequency in list to scale colors
    const maxVal = Math.max(...wordList.map(w => w.size));
    const minVal = Math.min(...wordList.map(w => w.size));
    const diff = maxVal - minVal || 1;

    wordList.forEach(w => {
        const span = document.createElement("span");
        span.className = "cloud-word";
        span.textContent = w.text;
        
        // Normalize size to font-size range: 12px to 38px
        const norm = (w.size - minVal) / diff;
        const fontSize = 12 + norm * 26;
        span.style.fontSize = `${fontSize}px`;
        
        // Color gradient: transition from soft indigo to vibrant emerald green
        const hue = 120 + norm * 80; // color spectrum shift
        span.style.color = `hsl(${hue}, 70%, 65%)`;

        container.appendChild(span);
    });
}

// OCR Processing with Tesseract.js
async function handleOcrFile(file) {
    if (!file || !file.type.startsWith("image/")) {
        alert("Please upload a valid image file (PNG, JPG, or JPEG).");
        return;
    }

    const overlay = document.getElementById("ocr-progress-overlay");
    const statusText = document.getElementById("ocr-status");
    const progressBar = document.getElementById("ocr-progress-bar");
    const percentageText = document.getElementById("ocr-percentage");

    // Reset progress UI
    progressBar.style.width = "0%";
    percentageText.textContent = "0%";
    statusText.textContent = "Initializing OCR Engine...";
    overlay.style.display = "flex";

    try {
        const result = await Tesseract.recognize(
            file,
            'eng',
            {
                logger: m => {
                    if (m && m.status) {
                        let statusMsg = m.status;
                        if (statusMsg === 'recognizing text') {
                            statusMsg = 'Extracting text...';
                        } else if (statusMsg === 'loading tesseract core') {
                            statusMsg = 'Loading AI Core...';
                        } else if (statusMsg === 'initializing api') {
                            statusMsg = 'Initializing AI API...';
                        } else if (statusMsg === 'loading language traineddata') {
                            statusMsg = 'Loading Language Data...';
                        }
                        
                        statusText.textContent = statusMsg.charAt(0).toUpperCase() + statusMsg.slice(1);
                        
                        if (typeof m.progress === 'number') {
                            const pct = Math.round(m.progress * 100);
                            progressBar.style.width = `${pct}%`;
                            percentageText.textContent = `${pct}%`;
                        }
                    }
                }
            }
        );

        const extractedText = result.data.text.trim();
        if (extractedText) {
            const inputField = document.getElementById("review-input");
            inputField.value = extractedText;
            
            // Visual success indicator: flash green border
            inputField.style.borderColor = "var(--color-positive)";
            setTimeout(() => {
                inputField.style.borderColor = "";
            }, 1000);
            
            // Auto run analysis on extracted text
            await performAnalysis();
        } else {
            alert("No text could be identified in the uploaded image. Please make sure it is a high-contrast text screenshot or photo.");
        }
    } catch (err) {
        console.error("OCR Failure: ", err);
        alert("Failed to extract text: " + err.message);
    } finally {
        overlay.style.display = "none";
    }
}

// === Live Camera Facial Expression Tracking ===
let isCameraRunning = false;
let webcamStream = null;
let cameraIntervalId = null;
let modelsLoaded = false;

// Async function to load models from static directory
async function loadFaceModels() {
    try {
        console.log("Loading face detection & expression models...");
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
        await faceapi.nets.faceExpressionNet.loadFromUri('/models');
        modelsLoaded = true;
        console.log("Face expression models loaded successfully.");
    } catch (err) {
        console.error("Failed to load face expression models:", err);
    }
}

async function toggleCamera() {
    if (isCameraRunning) {
        stopCamera();
    } else {
        await startCamera();
    }
}

async function startCamera() {
    if (!modelsLoaded) {
        alert("Face expression models are still loading. Please try again in a few seconds.");
        return;
    }

    const video = document.getElementById("webcam");
    const placeholder = document.getElementById("camera-placeholder");
    const toggleBtn = document.getElementById("camera-toggle-btn");
    const placeholderText = document.getElementById("camera-placeholder-text");

    placeholderText.textContent = "Requesting camera access...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 320 },
                height: { ideal: 240 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        webcamStream = stream;
        isCameraRunning = true;

        // Toggle layout to split-screen
        document.querySelector(".container").classList.add("camera-mode");

        // Animate placeholder out
        placeholder.style.opacity = "0";
        setTimeout(() => {
            if (isCameraRunning) placeholder.style.display = "none";
        }, 300);

        toggleBtn.textContent = "Stop Camera";
        toggleBtn.classList.add("active");

        // Start tracking when metadata is loaded
        video.onloadedmetadata = () => {
            video.play();
            startExpressionTrackingLoop(video);
        };
    } catch (err) {
        console.error("Camera access failed:", err);
        alert("Could not access camera. Please make sure webcam permissions are granted in your browser.");
        placeholderText.textContent = "Camera Offline";
    }
}

function stopCamera() {
    isCameraRunning = false;

    // Revert layout back to standard full width
    document.querySelector(".container").classList.remove("camera-mode");

    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }

    if (cameraIntervalId) {
        clearInterval(cameraIntervalId);
        cameraIntervalId = null;
    }

    const video = document.getElementById("webcam");
    video.srcObject = null;

    const canvas = document.getElementById("camera-overlay");
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const placeholder = document.getElementById("camera-placeholder");
    const toggleBtn = document.getElementById("camera-toggle-btn");
    const placeholderText = document.getElementById("camera-placeholder-text");

    placeholder.style.display = "flex";
    setTimeout(() => {
        placeholder.style.opacity = "1";
    }, 50);

    placeholderText.textContent = "Camera Offline";
    toggleBtn.textContent = "Start Camera";
    toggleBtn.classList.remove("active");

    updateFaceUI(null);
}

function startExpressionTrackingLoop(video) {
    const canvas = document.getElementById("camera-overlay");
    const displaySize = { width: video.videoWidth || 320, height: video.videoHeight || 240 };
    
    canvas.width = displaySize.width;
    canvas.height = displaySize.height;
    faceapi.matchDimensions(canvas, displaySize);

    cameraIntervalId = setInterval(async () => {
        if (!isCameraRunning) return;

        try {
            // Tiny Face Detector uses less CPU
            const detection = await faceapi
                .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 128, scoreThreshold: 0.5 }))
                .withFaceExpressions();

            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (detection) {
                const resizedDetection = faceapi.resizeResults(detection, displaySize);
                const box = resizedDetection.detection.box;

                // Draw Indigo Border Box
                ctx.strokeStyle = "rgba(99, 102, 241, 0.85)";
                ctx.lineWidth = 3;
                ctx.strokeRect(box.x, box.y, box.width, box.height);

                // Sort emotions by confidence
                const expressions = resizedDetection.expressions;
                const sorted = Object.entries(expressions).sort((a, b) => b[1] - a[1]);
                const dominantExpression = sorted[0][0];
                const dominantConfidence = sorted[0][1];

                // Draw Text Label above the box
                ctx.fillStyle = "rgba(99, 102, 241, 0.95)";
                ctx.font = "bold 13px Outfit, sans-serif";
                const labelText = `${dominantExpression.toUpperCase()} ${(dominantConfidence * 100).toFixed(0)}%`;
                const textWidth = ctx.measureText(labelText).width;
                ctx.fillRect(box.x, box.y - 22, textWidth + 12, 20);

                ctx.fillStyle = "#ffffff";
                ctx.fillText(labelText, box.x + 6, box.y - 7);

                // Update results card
                updateFaceUI({ expression: dominantExpression, confidence: dominantConfidence });
            } else {
                updateFaceUI(null);
            }
        } catch (err) {
            console.error("Tracking error:", err);
        }
    }, 150); // 150ms interval ~ 6 FPS (smooth but CPU lightweight)
}

function updateFaceUI(data) {
    const labelEl = document.getElementById("face-label");
    const scoreEl = document.getElementById("face-score");
    const probEl = document.getElementById("face-prob");
    const posGauge = document.getElementById("face-gauge-fill-pos");
    const negGauge = document.getElementById("face-gauge-fill-neg");

    if (!data) {
        labelEl.textContent = "-";
        labelEl.style.backgroundColor = "var(--color-neutral)";
        scoreEl.textContent = "+0.000";
        scoreEl.style.color = "var(--text-secondary)";
        probEl.textContent = "0.0%";
        posGauge.style.width = "0%";
        negGauge.style.width = "0%";
        latestScores.face = 0.0;
        updateUnifiedUI();
        return;
    }

    const { expression, confidence } = data;
    let score = 0.0;
    let label = expression.charAt(0).toUpperCase() + expression.slice(1);
    let color = "var(--color-neutral)";

    switch (expression) {
        case "happy":
            score = 1.0 * confidence;   // ⬆ MAX positive — full 1.0 weight
            color = "var(--color-positive)";
            break;
        case "surprised":
            score = 0.6 * confidence;   // ⬆ raised from 0.4 → 0.6
            color = "var(--primary-purple)";
            break;
        case "neutral":
        default:
            score = 0.05 * confidence;  // slight positive nudge for calm/neutral
            color = "var(--color-positive)";  // green — neutral treated as positive
            label = "Neutral";
            break;
        case "sad":
            score = -0.6 * confidence;
            color = "var(--color-negative)";
            break;
        case "angry":
            score = -0.9 * confidence;
            color = "var(--color-negative)";
            break;
        case "fearful":
            score = -0.5 * confidence;
            color = "var(--primary-pink)";
            break;
        case "disgusted":
            score = -0.8 * confidence;
            color = "var(--color-negative)";
            break;
    }

    labelEl.textContent = label;
    labelEl.style.backgroundColor = color;

    scoreEl.textContent = (score >= 0 ? "+" : "") + score.toFixed(3);
    scoreEl.style.color = color;

    probEl.textContent = (confidence * 100).toFixed(1) + "%";

    if (score >= 0) {
        // Positive: scale so happy=1.0 → 100% bar fill
        posGauge.style.width = Math.min(score * 100, 100) + "%";
        negGauge.style.width = "0%";
    } else {
        // Negative: scale similarly
        negGauge.style.width = Math.min(Math.abs(score) * 100, 100) + "%";
        posGauge.style.width = "0%";
    }

    // Save latest face score and trigger unified scoring
    latestScores.face = score;
    updateUnifiedUI();
}

// Update Unified Multimodal Favorability Score Card
function updateUnifiedUI() {
    const labelEl = document.getElementById("unified-label");
    const scoreEl = document.getElementById("unified-score");
    const posGauge = document.getElementById("unified-gauge-fill-pos");
    const negGauge = document.getElementById("unified-gauge-fill-neg");
    const textWeightEl = document.getElementById("unified-text-weight");
    const faceWeightEl = document.getElementById("unified-face-weight");

    let unifiedScore = 0.0;

    if (isCameraRunning) {
        // Webcam is online: Text is weighted 60% (30% LR, 30% DB), Face is weighted 40%
        unifiedScore = 0.3 * latestScores.lr + 0.3 * latestScores.db + 0.4 * latestScores.face;
        
        textWeightEl.textContent = "Text Weight: 60%";
        faceWeightEl.textContent = "Face Weight: 40%";
        faceWeightEl.style.display = "inline-block";
    } else {
        // Webcam is offline: Text is weighted 100% (50% LR, 50% DB)
        unifiedScore = 0.5 * latestScores.lr + 0.5 * latestScores.db;
        
        textWeightEl.textContent = "Text Weight: 100%";
        faceWeightEl.style.display = "none";
    }

    // Classify unified sentiment label and color
    let label = "Neutral";
    let color = "var(--color-neutral)";

    if (unifiedScore > 0.15) {
        label = "Favorable";
        color = "var(--color-positive)";
    } else if (unifiedScore < -0.15) {
        label = "Unfavorable";
        color = "var(--color-negative)";
    }

    // Update UI elements
    labelEl.textContent = label;
    labelEl.style.backgroundColor = color;

    scoreEl.textContent = (unifiedScore >= 0 ? "+" : "") + unifiedScore.toFixed(3);
    scoreEl.style.color = color;

    if (unifiedScore >= 0) {
        posGauge.style.width = (unifiedScore * 100) / 2 + "%";
        negGauge.style.width = "0%";
    } else {
        negGauge.style.width = (Math.abs(unifiedScore) * 100) / 2 + "%";
        posGauge.style.width = "0%";
    }
}



