import pandas as pd

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class FeatureEngineering():
    
    def __init__(self):
        
        self.text_column = 'description'
        self.categorical_columns = [
            'type',
            'country',
            'rating',
            'platform'
        ]
        self.numerical_columns = [
            'duration',
            'year_added'
        ]

    def date_feature(self,df):

        df = df.copy()

        # convert to datetime and extract year
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
        df["year_added"] = df["date_added"].dt.year

        # fill null values with mode
        df["year_added"].fillna(df["year_added"].mode()[0], inplace=True)
        
        return df

    def duration_feature(self,df):

        df = df.copy()

        # extract number from duration
        df["duration"] = df["duration"].str.extract("(\d+)").astype(float)

        return df
    
    def apply_feature_engineering(self,df):

        df = self.date_feature(df)
        df = self.duration_feature(df)

        return df

    def create_preprocessor(self):

        tfidf  = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1,2)
            )
        
        categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers= [
                ('text',tfidf,self.text_column),
                ('cat',categorical_pipeline,self.categorical_columns),
                ('num',numerical_pipeline,self.numerical_columns)
            ]
        )

        return preprocessor