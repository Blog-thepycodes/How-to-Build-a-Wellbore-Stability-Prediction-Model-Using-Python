import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox


# Initialize the root Tkinter window
root = tk.Tk()
root.title("Wellbore Stability Prediction - The Pycodes")
root.geometry("400x300")


# Initialize empty lists for labels and entries
labels = []
entries = []


# Let us Create labels and entry widgets for each feature
feature_names = ["Depth", "GammaRay", "Resistivity", "Density", "Sonic"]
for i, feature in enumerate(feature_names):
   label = tk.Label(root, text=feature)
   label.grid(row=i, column=0)
   labels.append(label)


   entry = tk.Entry(root)
   entry.grid(row=i, column=1)
   entries.append(entry)


def load_data():
   global df, features, target, X_train, X_test, y_train, y_test, scaler, poly, selector, stacking_model, best_gb_model, best_rf_model


   file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
   if not file_path:
       return


   df = pd.read_csv(file_path)


   # Define the features and the target
   features = df.drop(columns=['WellboreStability'])
   target = df['WellboreStability']


   # Handle class imbalance with SMOTE
   smote = SMOTE(random_state=42)
   features_res, target_res = smote.fit_resample(features, target)


   # Feature Engineering: Adding Polynomial Features
   poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
   features_poly = poly.fit_transform(features_res)


   # Feature Selection
   selector = SelectKBest(score_func=f_classif, k=20)
   features_selected = selector.fit_transform(features_poly, target_res)


   # Now We Split the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(features_selected, target_res, test_size=0.2, random_state=42)


   # Standardize the features
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)


   # Initialize models
   gb_model = GradientBoostingClassifier(random_state=42)
   rf_model = RandomForestClassifier(random_state=42)
   lr_model = LogisticRegression(random_state=42)


   # Hyperparameter tuning using Grid Search
   param_grid_gb = {
       'n_estimators': [100, 200],
       'learning_rate': [0.01, 0.1],
       'max_depth': [3, 5]
   }


   param_grid_rf = {
       'n_estimators': [100, 200],
       'max_depth': [None, 10],
       'min_samples_split': [2, 5]
   }


   grid_search_gb = GridSearchCV(gb_model, param_grid_gb, cv=5, scoring='accuracy')
   grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')


   grid_search_gb.fit(X_train, y_train)
   grid_search_rf.fit(X_train, y_train)


   best_gb_model = grid_search_gb.best_estimator_
   best_rf_model = grid_search_rf.best_estimator_


   # Ensemble model using Stacking Classifier
   stacking_model = StackingClassifier(estimators=[
       ('gb', best_gb_model),
       ('rf', best_rf_model)
   ], final_estimator=lr_model, cv=5)


   # Train the ensemble model
   stacking_model.fit(X_train, y_train)


   messagebox.showinfo("Data Load", "Data loaded and model trained successfully!")


def predict():
   try:
       input_data = [float(entry.get()) for entry in entries]
       input_poly = poly.transform([input_data])
       input_selected = selector.transform(input_poly)
       input_scaled = scaler.transform(input_selected)


       prediction = stacking_model.predict(input_scaled)
       result = "Stable" if prediction[0] == 1 else "Unstable"


       messagebox.showinfo("Prediction Result", f"This wellbore is predicted to be: {result}")
   except ValueError:
       messagebox.showerror("Input Error", "Please enter valid numeric values for all features.")
   except NameError:
       messagebox.showerror("Model Error", "Please load the data first and train the model.")


def show_confusion_matrix():
   if 'stacking_model' in globals():
       y_pred = stacking_model.predict(X_test)
       conf_matrix = confusion_matrix(y_test, y_pred)
       plt.figure(figsize=(10, 6))
       sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stable', 'Unstable'], yticklabels=['Stable', 'Unstable'])
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.title('Confusion Matrix for Wellbore Stability Prediction')
       plt.show()


def show_roc_curve():
   if 'stacking_model' in globals():
       y_pred_proba = stacking_model.predict_proba(X_test)[:, 1]
       precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
       plt.figure(figsize=(10, 6))
       plt.plot(recall, precision, marker='.')
       plt.xlabel('Recall')
       plt.ylabel('Precision')
       plt.title('Precision-Recall Curve')
       plt.show()


def show_classification_report():
   if 'stacking_model' in globals():
       y_pred = stacking_model.predict(X_test)
       report = classification_report(y_test, y_pred, target_names=['Unstable', 'Stable'])
       messagebox.showinfo("Classification Report", report)


def show_feature_importance():
   if 'stacking_model' in globals():
       feature_importances = best_gb_model.feature_importances_
       features_list = selector.get_feature_names_out(poly.get_feature_names_out(features.columns))
       plt.figure(figsize=(12, 8))
       sns.barplot(x=feature_importances, y=features_list)
       plt.xlabel('Feature Importance')
       plt.ylabel('Feature')
       plt.title('Feature Importance in Predicting Wellbore Stability')
       plt.show()


# Create GUI components
load_button = tk.Button(root, text="Load Data", command=load_data)
load_button.grid(row=len(feature_names), column=0, columnspan=2)


predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.grid(row=len(feature_names) + 1, column=0, columnspan=2)


conf_matrix_button = tk.Button(root, text="Show Confusion Matrix", command=show_confusion_matrix)
conf_matrix_button.grid(row=len(feature_names) + 2, column=0, columnspan=2)


roc_curve_button = tk.Button(root, text="Show ROC Curve", command=show_roc_curve)
roc_curve_button.grid(row=len(feature_names) + 3, column=0, columnspan=2)


classification_report_button = tk.Button(root, text="Show Classification Report", command=show_classification_report)
classification_report_button.grid(row=len(feature_names) + 4, column=0, columnspan=2)


feature_importance_button = tk.Button(root, text="Show Feature Importance", command=show_feature_importance)
feature_importance_button.grid(row=len(feature_names) + 5, column=0, columnspan=2)


root.mainloop()
