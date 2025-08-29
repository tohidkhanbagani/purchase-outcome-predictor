# Purchase Outcome Prediction System

This repository contains a machine learning project that predicts customer purchase outcomes (Keep, Exchange, or Refund). The system is built with Python, FastAPI, and XGBoost, and includes an interactive frontend for real-time simulations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Business Impact](#business-impact)
- [Technical Impact](#technical-impact)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Pipeline Diagram](#pipeline-diagram)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

## Overview

This project provides a predictive model that analyzes customer and product data to forecast whether a purchase will be kept, exchanged, or refunded. The system is exposed via a FastAPI backend and includes a user-friendly frontend for single and bulk predictions.

## Features

*   **Purchase Outcome Prediction:** Predicts the outcome of a purchase (Keep, Exchange, Refund) based on customer behavior and product information.
*   **Bulk Prediction:** Supports bulk predictions for multiple purchase scenarios.
*   **Interactive Frontend:** A web-based interface for simulating purchase outcomes and visualizing results.
*   **API Endpoints:** A robust FastAPI backend with well-defined endpoints for predictions and data retrieval.
*   **Modular Design:** A clear and organized project structure that separates concerns for data processing, model training, and API development.

## Business Impact

*   **Proactive Customer Service:** Identify customers likely to return or exchange products and offer proactive support.
*   **Inventory Management:** Optimize stock levels by anticipating returns and exchanges.
*   **Reduced Return Rates:** Understand the factors that lead to returns and take steps to mitigate them.
*   **Enhanced Customer Segmentation:** Segment customers based on their predicted purchase behavior.

## Technical Impact

*   **Scalable and High-Performance:** The FastAPI backend ensures high performance and scalability.
*   **Maintainable Codebase:** The modular architecture makes the project easy to maintain and extend.
*   **Reproducible Results:** The use of a structured training pipeline ensures that the model's results are reproducible.
*   **Easy Integration:** The RESTful API allows for easy integration with other systems and applications.

## Architecture

The project is organized into the following directories:

*   `api/`: Contains the FastAPI application, including the main API script and input class definitions.
*   `data/`: Holds the raw, processed, and encoded data used in the project.
*   `frontend/`: Contains the HTML file for the interactive frontend.
*   `models/`: Stores the trained machine learning models and data processing objects.
*   `notebook/`: Jupyter notebooks for data exploration and model development.
*   `src/`: The core source code, including:
    *   `data_ingestion/`: Scripts for data loading and validation.
    *   `features/`: Scripts for feature engineering and data preprocessing.
    *   `models/`: The purchase outcome prediction model.
    *   `pipelines/`: The training and inference pipelines.
    *   `utils/`: Utility functions used throughout the project.

## Folder Structure
```
purchase prediction/
├── api/
│   ├── input_classes.py
│   └── main.py
├── data/
│   ├── processed/
│   └── raw/
├── frontend/
│   └── predictive_outcome_simulator.html
├── models/
│   ├── data_processing_models/
│   ├── grid_search_models/
│   └── trained_models/
├── notebook/
│   ├── data_exploration.ipynb
│   └── predictive_model.ipynb
├── src/
│   ├── data_ingestion/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── utils/
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## Pipeline Diagram
```
+-----------------+      +----------------------+      +--------------------+
|                 |      |                      |      |                    |
|   Raw Data      +----->+  Data Ingestion      +----->+  Feature           |
|   (.csv)        |      |   (Ingest Class)     |      |  Engineering       |
|                 |      |                      |      | (InputModelFeatures|
+-----------------+      +----------------------+      |     Class)         |
                                                       |                    |
                                                       +----------+---------+
                                                                  |
                                                                  |
                                                                  v
+-----------------+      +----------------------+      +----------+---------+
|                 |      |                      |      |                    |
|   User Input    +----->+  FastAPI Backend     +----->+  Model Inference   |
| (JSON)          |      |   (main.py)          |      | (ModelInference    |
|                 |      |                      |      |     Class)         |
+-----------------+      +----------------------+      +----------+---------+
                                                                  |
                                                                  |
                                                                  v
+-----------------+      +----------------------+      +----------+---------+
|                 |      |                      |      |                    |
|   Prediction    <-----+  Purchase Outcome    <-----+  Encoded & Scaled   |
|  (JSON)         |      |   Model              |      |   Data             |
|                 |      | (XGBClassifier)      |      |                    |
+-----------------+      +----------------------+      +--------------------+
```

## Getting Started

To set up and run the project locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tohidkhanbagani/purchase-outcome-predictor
    cd "purchase prediction"
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv env
    # On Windows
    .\env\Scripts\activate
    # On macOS/Linux
    source env/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the FastAPI application:**
    ```bash
    uvicorn api.main:app --reload
    ```
    The API will be accessible at `http://127.0.0.1:8000`. You can view the interactive API documentation at `http://127.0.0.1:8000/docs`.

5.  **Open the frontend:**
    Open the `frontend/predictive_outcome_simulator.html` file in your web browser to interact with the application.

## Dependencies

The project relies on the following key Python libraries:

*   `fastapi`: For building the web API.
*   `uvicorn`: For running the FastAPI application.
*   `pandas`: For data manipulation and analysis.
*   `numpy`: For numerical operations.
*   `scikit-learn`: For machine learning utilities.
*   `xgboost`: For the prediction model.
*   `pydantic`: For data validation.
*   `joblib`: For model serialization.