# Intelligent Reading Comprehension and Quiz Generation System

## Overview
This project is an AI-powered Reading Comprehension and Quiz Generation System built using the RACE (ReAding Comprehension from Examinations) dataset. 

The system automatically generates comprehension questions, predicts correct answers, creates distractor options, evaluates user responses, and provides hints.

## System Architecture
The project is structured into three main 

1. **Data Layer**: Handles the RACE dataset loading, preprocessing, and feature engineering (including One-Hot Encoding and TF-IDF vectorization).

2. **Model Layer**: 
    * **Model A (Q&A Generator / Verifier)**: A pipeline utilizing traditional ML (Logistic Regression, SVM) and unsupervised learning (K-Means/Label Propagation) to verify answers and generate templates.

    * **Model B (Distractor & Hint Generator)**: Uses ML ranking and cosine similarity to generate plausible, incorrect distractor options and extract graduated hints for the user.
    
3. **UI Layer**: An interactive user interface built to wire both models together for a seamless user experience.

## Tech Stack
* **Language:** Python 
* **Machine Learning:** scikit-learn, pandas, numpy 
* **Frontend UI:** Streamlit 