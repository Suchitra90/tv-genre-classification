import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs('outputs',exist_ok=True)

pipeline = joblib.load('outputs/model.pkl')
X_test = joblib.load('outputs/X_test.pkl')

X_sample = X_test.sample(100,random_state=42)

feature_pipeline = pipeline.named_steps['features']
model = pipeline.named_steps['model']
class_names = model.classes_

X_transformed = feature_pipeline.transform(X_sample)
feature_names = feature_pipeline.get_feature_names_out()

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

explainer = shap.Explainer(model,X_transformed)
shap_values = explainer(X_transformed)

plt.figure()
shap.summary_plot(shap_values,X_transformed,feature_names=feature_names,class_names=class_names,show=False)
plt.savefig('plots/shap_summary_plot.png',bbox_inches='tight')

print('SHAP explainability plots saved in plots folder')