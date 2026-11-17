# Predicting Mortality in Septic ICU Patients

# Overview

This project explores the use of machine learning to predict mortality in patients admitted to the Intensive Care Unit (ICU) with sepsis using rapidly obtainable clinical variables. The project uses clinical data to train and evaluate predictive models, with a primary focus on a Random Forest classifier. The pipeline includes data loading, preprocessing, feature selection, model training, probability calibration, and performance evaluation. It also generates statistical reports and visualizations to assess model performance, including ROC curves, precision-recall curves, confusion matrices, calibration curves, and feature importance analyses. The ultimate goal is to investigate whether readily available clinical information can help identify patients at increased risk of mortality.

# Features

Data preprocessing: Handles clinical data preparation, feature encoding, scaling, and feature selection.
Machine learning: Trains classification models, with Random Forest as the primary model.
Model evaluation: Evaluates performance using metrics such as accuracy, ROC-AUC, precision-recall AUC, Brier score, and log loss.
Probability calibration: Uses isotonic regression to improve the reliability of predicted probabilities.
Statistical analysis: Supports cross-validation and bootstrap confidence intervals.
Visualization: Generates performance plots and other diagnostic visualizations.
Reporting: Saves model evaluation results, predictions, and analysis reports.



# Installation and Setup

Follow these instructions to download the repository and run the project on your local computer.

1. Install prerequisites
Make sure you have the following installed:
Python 3.10 or newer
Git
pip (Python package installer)

   You can verify your Python installation by running the following command:
python --version

2. Clone the repository
Open a terminal (or command prompt) and run:
git clone https://github.com/GrandhiA/Aditya-Grandhi-JEI.git

   Navigate into the project directory:
cd Aditya-Grandhi-JEI

3. Create a virtual environment
It is recommended to use a virtual environment to keep project dependencies separate from other Python projects.

   Mac/Linux:
python3 -m venv venv
source venv/bin/activate



   Windows:
python -m venv venv
venv\\Scripts\\activate

4. Install dependencies
Install the Python libraries used by the machine learning pipeline:

   pip install numpy pandas scikit-learn matplotlib seaborn shap optuna openpyxl joblib

5. Add the dataset
Place the required clinical dataset in the root project directory.
The current main.py configuration expects a CSV file named "New\_set.csv"

   The dataset should contain a target column named Mortality, which represents the patient mortality outcome.
Note: The dataset is not included in the repository. You must obtain the appropriate dataset separately and ensure that its column names and structure match the project's configuration.

6. Run the project
Once the dependencies are installed and the dataset is in place, execute:
python main.py

   The program will load the dataset, preprocess the clinical variables, train the machine learning model, evaluate its performance, and generate reports and visualizations.



# Project Structure

The repository is organized into several Python modules:
File                   Description
=====                  ===========
main.py                Main entry point that runs the machine learning pipeline.
config.py              Contains configuration settings, model parameters, and evaluation options.
data\_loader.py         Loads the clinical dataset and separates features from the target variable.
encoding.py            Handles categorical variable encoding.
evaluation.py          Evaluates model performance.
explainability.py      Calculate SHAP values
feature\_selection.py   Performs feature selection.
model.py               Defines and creates machine learning models.
optuna\_rf.py           Supports Random Forest hyperparameter optimization.
plotting.py            Generates performance visualizations.
preprocessing.py       Handles preprocessing, encoding, and scaling.
reporting.py           Creates evaluation reports and comparison tables.
threshold\_analysis.py  Performs prediction threshold analysis.
utils.py               Contains shared utility functions.



# Configuration

Project settings can be adjusted in config.py, including:

* Dataset and target column names.
* Training and testing split (default: 80% training, 20% testing).
* Random seed.
* Feature scaling and encoding options.
* Model parameters.
* Cross-validation settings.
* Bootstrap confidence intervals.
* Feature importance and explainability settings.
* Output directory and visualization settings.



# Output

After running the pipeline, the project generates model evaluation results, statistical reports, and visualizations.
The trained and calibrated model is also saved as "sepsis\_dashboard.pkl"
This file contains the trained model, scaler, feature names, and model performance metadata.



# Important Disclaimer

This project is intended for educational and research purposes. Its predictions should not be used as a substitute for professional medical judgment, clinical assessment, or validated medical decision-making systems. Any clinical application would require additional validation using appropriate patient populations and clinical oversight.

