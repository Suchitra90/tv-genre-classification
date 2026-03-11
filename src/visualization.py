import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datapreprocessing import Data_preprocessor

os.makedirs('outputs',exist_ok=True)

class visualiztaion():

    def __init__(self,data_path):
        
        self.data_path = data_path
        self.df = None

    def load_data(self):

        self.df = pd.read_csv(self.data_path)
        return self.df
    
    def preprocess(self):
        
        preprocessor = Data_preprocessor()
        self.df = preprocessor.preprocess(self.df)
        return self.df

    def genre_distribution_plot(self):

        plt.figure(figsize=(10,6))

        sns.countplot(
            data=self.df,
            x="genre",
            order=self.df["genre"].value_counts().index
        )

        plt.title("Genre Distribution")
        plt.xlabel("Genre")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig("plots/genre_distribution.png")
        plt.close()

        print("Saved: genre_distribution.png")

viz = visualiztaion('data/tv-shows.csv')
viz.load_data()
viz.preprocess()
viz.genre_distribution_plot()