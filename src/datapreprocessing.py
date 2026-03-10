import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Data_preprocessor():

    def __init__(self):
        pass

    def clean_data(self,df):

        df = df.copy()

        # remove duplicate values
        df.drop_duplicates(inplace=True)

        # Filling up the null values
        df["director"].fillna("Unknown", inplace=True)
        df["cast"].fillna("Unknown", inplace=True)
        df["country"].fillna("Unknown", inplace=True)
        df["rating"].fillna("Not Rated", inplace=True)
        df["description"].fillna("", inplace=True)

        return df

    def create_target_variable(self,df):

        df = df.copy()

        # Extract first genre
        df["genre"] = df["listed_in"].apply(lambda x: x.split(",")[0].strip())

        # consider top genres
        top_genres = df["genre"].value_counts().head(12).index
        df = df[df["genre"].isin(top_genres)]

        return df
    
    def preprocess(self,df):

        df = self.clean_data(df)
        df = self.create_target_variable(df)

        return df