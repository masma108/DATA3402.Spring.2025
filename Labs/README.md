 ![](UTA-DataScience-Logo.png)

 # Predicting Customer Churn

  **One Sentence Summary:** This project uses machine learning to predict which bank customers are likely to leave, using a dataset from the Kaggle playground series.

 ## Overview

 This project addresses the challenge of predicting customer churn, which involves determining which customers will stop using a bank's services. The approach taken involves analyzing a dataset with customer information, including demographics, account details, and service usage. The RandomForestClassifier model was employed to learn from this data, and it was able to predict customer churn with reasonable accuracy, identifying key factors that contribute to customers leaving the bank.

 ## Summary of Workdone

 ### Data

 The data used in this project is in CSV format and contains a mix of numerical features, such as age and balance, and categorical features, like gender and geography. The output is a binary flag indicating whether a customer churned or not.  The training data has 10,000 entries. The data was split into training and testing sets, with the training set used to train the model.

 #### Preprocessing / Clean up

 The data was checked for missing values, and there were none. Outliers were identified using the IQR method. Categorical features, specifically "Geography" and "Gender," were one-hot encoded to convert them into a numerical format suitable for the machine learning model.

 #### Data Visualization

 Histograms were used to visualize the distribution of numerical features for customers who churned and those who did not, helping to understand which numerical features might be important for prediction. Count plots were used to show the distribution of categorical features and their relationship to churn. Additionally, a correlation matrix was used to visualize the relationship between the numerical features.

 ### Problem Formulation

 The problem is formulated such that the input to the model is a set of customer features, both numerical and categorical, and the output is a prediction of whether a customer will churn (1) or not (0). The RandomForestClassifier model was used for this classification task.  RandomForest was chosen for its ability to handle both categorical and numerical data, its robustness to outliers, and its relatively good performance without extensive hyperparameter tuning. The key hyperparameters for RandomForest, such as the number of trees in the forest, the maximum depth of the trees, and the number of features considered when splitting a node, were set to their default values.

 ### Training

 The RandomForestClassifier model was trained using scikit-learn on a standard laptop. The training time was relatively quick, taking only a few seconds.  Random Forest training does not involve traditional training curves like loss vs epoch. The model is built by aggregating the results of many decision trees, and training was stopped after the trees were grown. No significant difficulties were encountered during training.

 ### Performance Comparison

 The key performance metric used was accuracy, which measures the proportion of correctly classified customers, whether they churned or not. The model achieved an accuracy of 81% on the test set.

 ### Conclusions

 The RandomForestClassifier model can effectively predict customer churn based on the given features.  Certain features, such as account balance and customer activity, appear to be more influential in predicting churn.

 ### Future Work

 Further exploration could involve trying other machine learning models, such as Gradient Boosting Machines or Support Vector Machines, to see if the performance can be improved.  More advanced feature engineering, such as creating new features or transforming existing ones, could also be explored. Analyzing the impact of different hyperparameter settings for the RandomForestClassifier could lead to better results.

 ## How to reproduce results

 To reproduce the results of this project, follow these steps: First, download the data from the Kaggle competition. Then, ensure that you have Python installed, along with the required libraries, including pandas and scikit-learn.  Finally, run the provided Jupyter Notebook, which contains the code for data loading, preprocessing, model training, and evaluation.

 ### Overview of files in repository

 The repository contains several key files.  README.md provides an overview of the project. Kaggle_Tabular_Data.ipynb is a Jupyter Notebook containing the code for data loading, preprocessing, model training, and evaluation.  train.csv is the training data, test.csv is the test data, and sample_submission.csv is the sample submission file for Kaggle.

 ### Software Setup

 To set up the software for this project, you will need Python and the following packages: pandas and scikit-learn.  Install Python (e.g., from python.org or using Anaconda), and then install the required packages using pip: `pip install pandas scikit-learn`.

 ### Data

 The data used in this project can be downloaded from the Kaggle competition page: [https://www.kaggle.com/competitions/playground-series-s4e1/overview](https://www.kaggle.com/competitions/playground-series-s4e1/overview).  The preprocessing steps are detailed in the Jupyter Notebook, including handling categorical variables and missing values.

 ### Training

 The model is trained by running the appropriate cells in the Jupyter Notebook. The notebook contains the code to train the RandomForestClassifier.

 #### Performance Evaluation

 Performance evaluation is also carried out within the Jupyter Notebook, using the scikit-learn library to calculate the accuracy of the model on the test data.
 
