# Product Attribute Extraction

## Overview

This project builds a machine learning system to extract multiple product attributes from text descriptions. It predicts attributes such as silhouette, fabric, neckline, sleeve, length, embellishment, color, and category using a trained model and improves results using rule-based logic.

## Features

* Multi-output prediction for 8 attributes
* Text vectorization using TF-IDF
* Random Forest model with MultiOutputClassifier
* Rule-based corrections for improved accuracy
* FastAPI endpoint for real-time inference

## Tech Stack

* Python
* Pandas
* Scikit-learn
* FastAPI
* Pickle

## Workflow

1. Load and clean dataset
2. Convert text into numerical vectors using TF-IDF
3. Train a multi-output Random Forest model
4. Evaluate model performance using accuracy and F1 score
5. Save model and vectorizer
6. Use FastAPI to serve predictions
7. Apply rule-based fixes to refine outputs

## Model Training

* Dataset: `dataset.csv`
* Input: product description
* Output: multiple attribute labels
* Data split: 80% training, 20% testing
* Evaluation metrics: accuracy and macro F1 score

Evaluation results are stored in:
`evaluation_results.txt`

## API Usage

### Endpoint

POST /extract

### Request

{
"description": "Red cotton maxi dress with lace details"
}

### Response

{
"attributes": {
"silhouette": "...",
"fabric": "cotton",
"neckline": "...",
"sleeve": "...",
"length": "long",
"embellishment": "lace",
"color": "...",
"category": "..."
}
}

## Project Structure

* api.py
* model.pkl
* vectorizer.pkl
* dataset.csv
* evaluation_results.txt

## Notes

* The system combines machine learning predictions with rule-based logic for better reliability.
* Unknown or missing values are handled using default labels.

## Future Improvements

* Improve model performance using advanced NLP models
* Increase dataset size and quality
* Add cloud deployment
* Build a frontend interface


