import argparse
import joblib 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os

from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from datapreprocessing import Data_preprocessor
from feature_engineering import FeatureEngineering

class Model_trainer():

    def __init__(self,data_path,model_name,seed):
        self.data_path = data_path
        self.model_name = model_name
        self.seed = seed

    def load_data(self):

        df = pd.read_csv(self.data_path)
        return df

    def split_data(self,df):

        X = df[[
            'description',
            'type',
            'country',
            'rating',
            'platform',
            'duration',
            'year_added'
            
        ]]

        y = df['genre']

        return train_test_split(X,y,test_size=0.2,random_state=self.seed)

    def get_model(self,model_name):

        if model_name == 'logistic':
            model =  LogisticRegression(max_iter=2000,random_state=self.seed)
            param_grid = {
                'model__C':[0.01,0,1,1,10]
            }

        elif model_name == 'rf':
            model = RandomForestClassifier(random_state=self.seed)

            param_grid = {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4]
            }
        
        elif model_name == 'lgbm':
            model = LGBMClassifier(random_state=self.seed)

            param_grid = {
                "model__n_estimators": [200, 300, 400],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [-1, 5, 10],
                "model__num_leaves": [31, 50, 80, 120],
                "model__min_child_samples": [10, 20, 30],
                "model__subsample": [0.7, 0.8, 1.0]
            }
        else:
            raise ValueError('Invalid Model Name')
        
        return model,param_grid
        
    def train_model(self):

        df = self.load_data()

        preprocessor = Data_preprocessor()
        df = preprocessor.preprocess(df)

        feature_engineering = FeatureEngineering()
        df = feature_engineering.apply_feature_engineering(df)
        
        feature_pipeline = feature_engineering.create_preprocessor()

        X_train,X_test, y_train, y_test = self.split_data(df)

        model,param_grid = self.get_model(self.model_name)

        # Save test data for evaluation
        joblib.dump(X_test, "outputs/X_test.pkl")
        joblib.dump(y_test, "outputs/y_test.pkl")

        pipeline = Pipeline(
            steps = [
                ('features',feature_pipeline),
                ('model', model)
            ]
        )

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=3,
            scoring="f1_weighted",
            random_state=self.seed,
            n_jobs=-1
        )

        os.makedirs("outputs", exist_ok=True)

        search.fit(X_train,y_train)

        best_model = search.best_estimator_

        joblib.dump(best_model, 'outputs/model.pkl')

        print('Best Parameters:',search.best_params_)

        print("Model trained and saved ")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',type=str,required=True)
    parser.add_argument('--model',type=str,default='logistic')
    parser.add_argument('--seed',type=int,default=42)

    args = parser.parse_args()

    trainer = Model_trainer(args.data_path,args.model,args.seed)

    trainer.train_model()