
#  Diabetes Prediction - GTC Machine Learning Internship Project 2

##  Overview
This notebook explores, builds, and evaluates machine learning models to predict the likelihood of diabetes in patients using health-related features such as glucose levels, BMI, age, blood pressure, and more.  

The project follows a complete ML pipeline:  
- Data exploration and visualization  
- Outlier handling and feature engineering  
- Preprocessing with imputation & scaling  
- Model training (Logistic Regression, SVM, Random Forest)  
- Model evaluation and comparison  
- Saving and loading the best model  
- Final prediction demo with sample patient data   



##  Project Structure

````
Diabetes_Prediction.ipynb   
diabetes.csv               
diabetes_model.pkl        
````


##  Dataset
The project uses the **Pima Indians Diabetes Dataset**, which contains diagnostic measurements for female patients aged ≥21.  

 Dataset link: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  

**Features include:**  
- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  
- Outcome (target: 1 = Diabetic, 0 = Non-Diabetic)  

---

##  Requirements
Install dependencies before running the notebook:  

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn 
````


##  How to Use

1. Open the notebook:

   ```
   Diabetes_Prediction.ipynb
   ```
2. Run all cells step by step to:

   * Explore and visualize the dataset
   * Train and evaluate models
   * Compare performances
   * Save/load the trained model
   * Test prediction on a sample patient

---

##  Results

* **Random Forest** achieved the best performance:

  * Test Accuracy: \~74.7%
  * Test F1 Score: \~0.68
* Logistic Regression and SVM served as reliable baselines with \~70–72% accuracy.

---

##  Key Features

* Outlier detection and treatment
* Feature engineering: Age groups, BMI categories, interaction terms
* Handling missing values (zero replacement + KNN imputation)
* Model tuning with GridSearchCV
* Visualizations: histograms, heatmaps, scatterplots, confusion matrices
* End-to-end prediction workflow with sample patient data

