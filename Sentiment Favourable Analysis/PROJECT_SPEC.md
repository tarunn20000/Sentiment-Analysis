# Project Specification: Sentiment Favorability Analysis (Java)

## Objective
To build a sentiment analysis project that captures **favorability** as a continuous score between `-1.0` (highly unfavorable) and `+1.0` (highly favorable), rather than a binary positive/negative classification.

## Pipeline Architecture (Java)
1. **Preprocessing (`Preprocessing.java`)**:
   - Clean HTML tags, URLs, numbers, and extra whitespaces.
   - Text normalization (lowercase).
   - Word tokenization and stopword removal.
   - Word stemming using a Porter Stemmer implementation.
2. **Feature Extraction (`Features.java`)**:
   - TF-IDF Sparse Vectorization.
   - Vocab creation and IDF weight calculation from a corpus.
3. **Models (`Model.java`)**:
   - Stochastic Gradient Descent (SGD) Logistic Regression model trained on TF-IDF features.
   - Feature weights (coefficients) extractor for explainability.
4. **Favorability Scoring Layer (`FavorabilityScore.java`)**:
   - Mathematical transformation of probability outputs to $[-1.0, 1.0]$.
   - Temperature scaling on implied logits to calibrate confidence.
5. **Evaluation (`Evaluate.java`)**:
   - Classification metrics: Accuracy, F1.
   - Regression metrics: MAE, MSE.
6. **Web Dashboard Backend (`App.java`)**:
   - Javalin-based server exposing REST API endpoints:
     - `/api/analyze`: Takes input text and returns predictions, word importances, and favorability scores.
     - `/api/trends`: Provides rolling average sentiment statistics for a sample corpus.
7. **Web Dashboard Frontend (`index.html`, `style.css`, `script.js`)**:
   - Premium glassmorphic interface showing sentiment gauges, word clouds (rendered via JavaScript), word impact bar charts, and rolling sentiment trends (rendered via Chart.js).

## Directory Layout
```
sentiment-favorability/
├── pom.xml
├── PROJECT_SPEC.md
├── .antigravity-rules
├── README.md
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── sentiment/
│   │   │           ├── Preprocessing.java
│   │   │           ├── Features.java
│   │   │           ├── Model.java
│   │   │           ├── FavorabilityScore.java
│   │   │           ├── Evaluate.java
│   │   │           └── App.java
│   │   └── resources/
│   │       └── public/
│   │           ├── index.html
│   │           ├── style.css
│   │           └── script.js
│   └── test/
│       └── java/
│           └── com/
│               └── sentiment/
│                   └── PipelineTest.java
```
