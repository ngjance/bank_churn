# Executive Summary

The goal of this project was to build two machine learning models, a classification model and a clustering model, to help a bank predict customer churn and understand the demographics of the customers who churned.

The data used in this project was collected from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset).

The classification model was evaluated using accuracy, precision, F1 score and AUC as the metrics. The results showed that the model had an accuracy of 84%, precision of 62%, F1 score of 59% and AUC score of 84%. This indicates that the model is able to correctly predict 84% of the customers who are likely to churn and has a 62% chance of correctly identifying a customer who will churn.

The clustering model was used to group customers based on their demographic characteristics. The results showed that there were four main demographic groups of customers who churned: (1) middle-class customers, (2) familyman, (3) working women and (4) customers who don't save aka spenders.

It should be noted that the results of this project may be limited by the quality and availability of the data used, as well as the assumptions made about the customers and their behavior. Additionally, there may be other factors not considered in this analysis that could impact customer churn.

Overall, this project provides valuable insights into the demographics of customers who are likely to churn, which can be used by the bank to improve its customer retention strategies.


# [Data Cleaning and EDA](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%201_Data_Cleaning_and_EDA.ipynb)
Link

# [Prepossesing and Modeling](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%202_Preprocessing_and_Modeling.ipynb)
Link

### Base Models
The base models include all the columns as features.
There shows signs of overfitting, hence we will reduce the features.

**Accuracy Score:**
|                    	| LogisticRegression 	|   KNN   	| Decision Tree 	| Random Forest 	| Bagging 	| AdaBoost 	|   SVC   	| XGBoost 	| LightGBM 	|
|:------------------:	|:------------------:	|:-------:	|:-------------:	|:-------------:	|:-------:	|:--------:	|:-------:	|:-------:	|:--------:	|
| Train Set Accuracy 	|            0.75574 	| 0.88224 	|       1.00000 	|       1.00000 	| 0.99166 	|  0.84943 	| 0.86580 	| 0.95957 	|  0.91787 	|
| Test Set Accuracy  	|            0.72450 	| 0.76700 	|       0.76150 	|       0.84100 	| 0.82150 	|  0.81850 	| 0.81600 	| 0.83050 	|  0.83800 	|


### Selecting the Features
The features we selected are:
estimated_salary, balance, age, credit_score, country x balance, products_number and gender

|                               	|     LogisticRegression 	|       KNN   	|     Decision Tree 	|     Random Forest 	|     Bagging 	|     AdaBoost 	|       SVC   	|     XGBoost 	|     LightGBM    	|
|:-----------------------------:	|:----------------------:	|:-----------:	|:-----------------:	|:-----------------:	|:-----------:	|:------------:	|:-----------:	|:-----------:	|:---------------:	|
|     Train Set Accuracy        	|                0.70477 	|     0.89648 	|           1.00000 	|           1.00000 	|     0.99245 	|      0.79452 	|     0.79610 	|     0.94989 	|      0.91284    	|
|     Test Set Accuracy         	|                0.71200 	|     0.74750 	|           0.77050 	|           0.82450 	|     0.82500 	|      0.78650 	|     0.78050 	|     0.84200 	|      0.84200    	|
|     Train Set Precision Score 	|                0.71438 	|     0.84855 	|           1.00000 	|           1.00000 	|     0.99667 	|      0.80528 	|     0.79826 	|     0.96549 	|         0.93073 	|
|     Test Set Precision Score  	|                0.37483 	|     0.41250 	|           0.41923 	|           0.55381 	|     0.51961 	|      0.47249 	|     0.46451 	|     0.60606 	|         0.59747 	|
|     Train Set F1 Score        	|                0.69800 	|     0.90314 	|           1.00000 	|           1.00000 	|     0.99242 	|      0.79084 	|     0.79536 	|     0.94904 	|      0.91099    	|
|     Test F1 Set F1 Score      	|                0.48754 	|     0.51113 	|           0.48600 	|           0.58264 	|     0.55357 	|      0.57765 	|     0.57829 	|     0.58201 	|      0.59898    	|
|     AUC                       	|                0.77000 	|     0.77000 	|           0.69000 	|           0.83000 	|     0.80000 	|      0.85000 	|     0.84000 	|     0.83000 	|      0.85000    	|

Based on the accuracy score, precision, F1 score and AUC, we streamlined to five models for further tuning:

1) Random Forest
2) SVC
3) AdaBoost
4) XGBoost
5) LightGBM



### PolynomialFeatures
We tried PolynomialFeatures and it did not significantly improve the models, hence we will not use it.



### GridSearch
Finally, we used GridSearch to find the best parameters that give the best precision score.
We are more concerned about precision score because 



### Final Models
The results after grid search is as below:

|                             	| Random Forest 	|   SVC   	| AdaBoost 	| XGBoost 	| LightGBM    	|
|:---------------------------:	|:-------------:	|:-------:	|:--------:	|:-------:	|:-----------:	|
| Train Set   Accuracy        	|    0.85022    	| 0.80404 	|  0.83795 	| 0.94989 	|   0.94470   	|
| Test Set   Accuracy         	|    0.80550    	|  0.7795 	|  0.80850 	| 0.84200 	|   0.84850   	|
| Train Set   Precision Score 	|    0.84945    	| 0.80428 	|  0.84512 	| 0.96549 	|   0.96268   	|
| Test Set   Precision Score  	|    0.50358    	| 0.46262 	|  0.50909 	| 0.60606 	|   0.62784   	|
| Train Set   F1 Score        	|    0.85039    	| 0.80397 	|  0.83625 	| 0.94904 	|   0.94360   	|
| Test F1   Set F1 Score      	|    0.59096    	| 0.57391 	|  0.59385 	| 0.58201 	|   0.59329   	|
| AUC                         	|    0.85000    	| 0.83000 	|  0.85000 	| 0.83000 	|  0.84000    	|

We recommend to use LightGBM as the final model given that it gives the highest accuracy score, precision score and F1 score.






# [Clustering](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%203_Clustering.ipynb)
Link


# Demo
Link



