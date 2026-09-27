# Sentiment Favorability Analysis Pipeline & Dashboard (Java)

This repository contains a modular Maven project for sentiment analysis with a **continuous favorability scoring system** ranging from `-1.0` (highly unfavorable) to `+1.0` (highly favorable).

By going beyond binary labels, this project calculates the model's confidence and maps it into a scale that represents nuance, accompanied by a Javalin web server serving a rich glassmorphic HTML/CSS/JS dashboard showing sentiment trends and key word explanations.

## Setup and Installation

1. **Prerequisites**: Make sure you have **Java 11+** and **Maven** installed and available on your PATH.
2. **Build and Install**:
   ```bash
   mvn clean install
   ```

## Running the Web Dashboard

To launch the Javalin server and open the web dashboard:
1. Run the main class:
   ```bash
   mvn exec:java -Dexec.mainClass="com.sentiment.App"
   ```
2. Open your web browser and navigate to:
   ```
   http://localhost:8080
   ```

## Running Tests

To run the unit test suite:
```bash
mvn test
```
