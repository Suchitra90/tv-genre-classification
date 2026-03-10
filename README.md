# TV Genre Classification using Machine Learning

## Project Overview:
    
    This project builds an machine learning pipeline to classify TV shows & Movies genres using metadata and textual descriptions. This pipeline includes data pre-processing, feature engineering, model training, hyperparameter tuning and model evaluation.

## Project Structure

```
tv-genre-classification
│
├── data/              # Dataset used for training
│   └── data.csv
│
├── src/               # Source code
│   ├── train.py       # Model training pipeline
│   ├── evaluate.py    # Model evaluation
│   ├── datapreprocessing.py
│   └── feature_engineering.py
|   |__ data_loader.py
│
├── outputs/           # Saved model and evaluation results
│   ├── model.pkl
│   ├── X_test.pkl
│   ├── y_test.pkl
│   └── metrics.json
│
├── plots/             # Generated plots
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```
## Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* LightGBM
* matplotlib
* joblib

## Setup Instructions

Install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Execution Instructions

### Train the Model

Run the following command to train the machine learning model:

```
python src/train.py --data_path data/data.csv --model lgbm --seed 42
```

This step performs:

* Data preprocessing
* Feature engineering
* Model training
* Hyperparameter tuning
* Saving the trained model

The trained model and test data will be saved in the **outputs/** folder.

---

### Evaluate the Model

Run the following command to evaluate the trained model:

```
python src/evaluate.py --model_path outputs/model.pkl --data_path data/tv-shows.csv
```

This step involves:

* Load the trained model
* Generate predictions on the test dataset
* Calculate evaluation metrics
* Generate evaluation plots

## Output Files

Running the project will generate:

* Trained model (`model.pkl`)
* Test dataset (`X_test.pkl`, `y_test.pkl`)
* Evaluation metrics (`metrics.json`)
* Confusion matrix plot
* Feature importance plot

The folders **outputs/** and **plots/** will be automatically created when the scripts are executed if they do not already exists

## Evaluation Metrics
Model performance is evaluated using
* Accuracy_score
* F1 score
* Classification report
* Confusion matrix