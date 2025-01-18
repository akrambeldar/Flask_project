
This repository contains a Flask web application that includes machine learning models and other assets. Please ensure you maintain the folder structure as described below to run the web app without any errors.

Directory Structure:
      --------------------
      flask_project/
      │
      ├── app/
      │   ├── __pycache__/                    
      │   ├── models/
      │   │   ├── fasttext_model.bin
      │   │   ├── logistic_regression_model.pkl
      │   │   └── tfidf_vectorizer.pkl
      │   ├── static/
      │   │   └── css/
      │   ├── templates/
      │   │   ├── classes/
      │   │   ├── browse/
      │   │   ├── categories/
      │   │   ├── clothes/
      │   │   ├── department/
      │   │   ├── division/
      │   │   ├── home/
      │   │   ├── product_retail/
      │   │   ├── reviews/
      │   │   └── search_results/
      │   ├── __init__.py
      │   └── routes.py
      │
      ├── cloth_data/                        # Dataset directory (if needed)
      ├── requirements.txt                   # Project dependencies
      └── run.py                              # Main entry point to run the Flask app

Prerequisites:
--------------
Before running the project, ensure that the required Python versions and dependencies are installed. The versions for each dependency are specified in the requirements.txt file.

Setting Up the Project:
------------------------
Install Python:
   Ensure you have Python installed. The project is compatible with the following versions:
   
  Flask==3.0.3
  pandas==2.2.2
  gensim==4.3.3
  scikit-learn==1.4.2
  numpy==1.26.4
  matplotlib==3.9.2
  seaborn==0.11.1
