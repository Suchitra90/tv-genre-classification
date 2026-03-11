import argparse
import joblib 
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score,classification_report,f1_score,confusion_matrix,ConfusionMatrixDisplay,precision_score,recall_score

class ModelEvaluator():

    def __init__(self,model_path,data_path):

        self.model_path = model_path
        self.data_path = data_path

    def load_data(self):

        df = pd.read_csv(self.data_path)
        return df
    
    def load_model(self):
         
         model = joblib.load(self.model_path)
         return model
    
    def load_test_data(self):

        X_test = joblib.load('outputs/X_test.pkl')
        y_test = joblib.load('outputs/y_test.pkl')
        return X_test,y_test
    
    def confusion_matrix_plot(self,y_test,predictions):

        os.makedirs('plots',exist_ok=True)

        cm = confusion_matrix(y_test,predictions)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(xticks_rotation=45)

        plt.title('Confusion Matrix')

        plt.savefig('plots/confusion_matrix.png')

        plt.close()

    def classificatio_report(self,y_test,predictions):

        report = classification_report(y_test, predictions, output_dict=True)

        report_df = pd.DataFrame(report).transpose()

        report_df.iloc[:-1, :-1].plot(kind="bar", figsize=(10,6))

        plt.title("Precision, Recall, F1 Score by Class")

        plt.xlabel("Class")

        plt.ylabel("Score")

        plt.tight_layout()

        plt.savefig("plots/classification_report.png")

        plt.close()

    def feature_importance_plot(self,model):

        os.makedirs("plots", exist_ok=True)

        model_step = model.named_steps["model"]

        if hasattr(model_step, "feature_importances_"):

            importances = model_step.feature_importances_

            indices = np.argsort(importances)[-20:]

            plt.figure()

            plt.barh(range(len(indices)), importances[indices])

            plt.yticks(range(len(indices)), indices)

            plt.title("Top Feature Importances")

            plt.savefig("plots/feature_importance.png")

            plt.close()
    
    def evaluate(self):

        model = self.load_model()

        X_test,y_test = self.load_test_data()

        predictions = model.predict(X_test)

        acc = accuracy_score(y_test,predictions)

        f1 = f1_score(y_test,predictions,average='weighted')

        precision = precision_score(y_test,predictions,average='weighted')

        recall = recall_score(y_test,predictions,average='weighted')

        metrics = {
            'accuracy score': acc,
            'f1_score' : f1,
            'precision score' : precision,
            'recall score' : recall
        }

        print(classification_report(y_test,predictions))

        with open('outputs/metrics.json','w') as f:
            json.dump(metrics,f)

        #plots
        self.confusion_matrix_plot(y_test,predictions)
        self.classificatio_report(y_test,predictions)
        self.feature_importance_plot(model)

        print('Evaluation Complete')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path',type=str)
    parser.add_argument('--data_path',type=str)

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model_path,args.data_path)

    evaluator.evaluate()



