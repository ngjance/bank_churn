# Executive Summary

The goal of this project is to help a bank predict which customers are likely to churn and understand the demographics of the customers who churned.
This will help the bank to improve their retention strategy and policy across the different functions such as marketing, products and customer service.

The data used in this project was collected from [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset).

First, a classification model was built and evaluated using accuracy, recall, F1 score and AUC as the metrics. The results showed that the model had an accuracy of 81%, recall of 72%, F1 score of 59% and AUC score of 85%. This indicates that the model is able to correctly predict 81% of the customers who are likely to churn and has a 72% chance of correctly identifying a customer who will churn.

Next, a clustering model was used to group customers based on their demographic characteristics. The results showed that there were four main demographic groups of customers who churned: (1) working males, (2) spenders, (3) working females and (4) young singles.

It should be noted that the results of this project may be limited by the quality and availability of the data used, as well as the assumptions made about the customers and their behavior. Additionally, there may be other factors not considered in this analysis that could impact customer churn.

Overall, this project provides valuable insights into the demographics of customers who are likely to churn, which can be used by the bank to improve its customer retention strategies.


# [Data Cleaning and EDA](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%201_Data_Cleaning_and_EDA.ipynb)
Using the free version of [atoti](https://www.atoti.io/), we ploted the below charts with ease to do a quick exploratory data analysis.

![image](https://user-images.githubusercontent.com/63915619/222766843-5d228b21-7074-4835-8ea3-a6c7575c9b4c.png)
Customers who churn are generally older.
<br><br>

![image](https://user-images.githubusercontent.com/63915619/222766922-c2af9c38-05e0-4e10-b694-bc02618cf0d2.png)
Customers who churn and don't churn have the similar credit score
<br><br>

![image](https://user-images.githubusercontent.com/63915619/222767951-b678aaed-1a5b-47d1-bb79-6235555699c4.png)
Females who churn earn slightly higher salary than females who don't churn.
Salary doesn't seem to be a contributing factor on why males churn.
<br><br>

![image](https://user-images.githubusercontent.com/63915619/222768114-5d2e42cb-5100-46a8-928c-eb7926e0f124.png)
Both females and male who churn have higher balances than those who don't churn.
Males also have higher balances compared to females.
<br><br>

![image](https://user-images.githubusercontent.com/63915619/222768349-6d2c6ef4-0411-4eb0-b6f5-f839c887fb97.png)
Both females and male who churn have slightly lower tenure than those who don't churn.
<br><br>

![image](https://user-images.githubusercontent.com/63915619/222768452-efaf0c6e-8e2d-4c91-8094-a881fa0450a4.png)
Males who churn hold the fewest number of products.
<br><br>

Note: The notebook is best viewed using Jupyter Lab.

# [Prepossesing and Modeling](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%202_Preprocessing_and_Modeling.ipynb)

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
<br><br>
Our goal is to reduce Type 2 error or False Negatives False negative in our case will be customers whom we predicted will not churn, but in reality churned.

We want to reduce such errors because if we predicted that a customer will not churn but in fact he / she churned, then we lost a customer.

In the opposite case however, where we predicted that a customer will churn and the customer actually did not churn, we would spend resources/effort to try to retain the customer which could have been redirected.

Hence, reducing type 2 error is more important as losing a customer cost more than spending resources on the wrong targeted customer.

In another words, we want to predict the customer who will churn as accurately as possible. This will help us to direct resources to the right customers and help retain the customer. Therefore, precision is an important metric in our model.
<br>

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
Finally, we used GridSearch to find the best parameters that give the best recall score followed by best accuracy score.

Below is the best parameters for each model:

| Model         	| Best Params for Recall Score                                                   	|
|---------------	|--------------------------------------------------------------------------------	|
| Random Forest 	| {'ccp_alpha': 0.001, 'criterion': 'entropy', 'max_depth': 30, 'n_estimators': 300}   	|
| SVC           	| {'C': 2, 'decision_function_shape': 'ovo', 'degree': 1, 'gamma': 'auto'}       	|
| Adaboost      	| {'learning_rate': 1, 'n_estimators': 400}                                      	|
| XGBoost       	| {'eta': 0.3, 'gamma': 0, 'max_depth': 6}                                     	|
| LightGBM      	| {'learning_rate': 0.1, 'max_depth': -1, 'n_estimators': 300, 'num_leaves': 31} 	|


### Final Models
The results after grid search is as below:

|                             	| Random Forest 	|   SVC   	| AdaBoost 	| XGBoost 	| LightGBM    	|
|:---------------------------:	|:-------------:	|:-------:	|:--------:	|:-------:	|:-----------:	|
| Train Set   Accuracy        	|    0.85022    	| 0.80404 	|  0.83795 	| 0.94989 	|   0.96554   	|
| Test Set   Accuracy         	|    0.80550    	|  0.7795 	|  0.80850 	| 0.84200 	|   0.83800   	|
| Train Set   Precision Score 	|    0.84945    	| 0.80428 	|  0.84512 	| 0.96549 	|   0.98051   	|
| Test Set   Precision Score  	|    0.50358    	| 0.46262 	|  0.50909 	| 0.60606 	|   0.59300   	|
| Train Set Recall Score      	|    0.85132    	| 0.80365 	|  0.82756 	| 0.93313 	|   0.94997   	|
| Test Set Recall Score       	|    0.71501    	| 0.75573 	|  0.71247 	| 0.55980 	|   0.55980   	|
| Train Set   F1 Score        	|    0.85039    	| 0.80397 	|  0.83625 	| 0.94904 	|   0.96500   	|
| Test F1   Set F1 Score      	|    0.59096    	| 0.57391 	|  0.59385 	| 0.58201 	|   0.57592   	|
| AUC                         	|    0.85000    	| 0.83000 	|  0.85000 	| 0.83000 	|  0.84000    	|

Given that our goal is to reduce type 2 error, recall score is the most important metric to us, followed by accuracy score.

SVC gives the highest recall score and the lowest accuracy score. SVC also has slow performance if the dataset is huge, which is probable in a bank.

Random Forest gives the second highest recall score and about 80.55% accuracy. Its AUC score is also higher than SVC. Thus, we will deploy Random Forest as our final model.

### Random Forest - Final Select Model
![image](https://user-images.githubusercontent.com/63915619/221340645-4aa06273-455b-4d25-8709-a377ad48ca4a.png)



### Shap Explainer
![image](https://user-images.githubusercontent.com/63915619/221340266-b2d9d8aa-30ba-478e-b6f8-ec40d32894e0.png)

Using Shap, we see that age is the most important contributing feature to the model, followed by product_number and country x balance.


![image](https://user-images.githubusercontent.com/63915619/221340307-81924c80-5daa-4a5b-a64d-beb49e9c29c3.png)

Taking one specific client to look at, we see that age, country x balance, and balance are pushing the model towards the base value and the client is 76% likely to churn.

Contrary, the number of products and gender features have negative contributing effect on him churning.


# [Clustering](https://github.com/ngjance/bank_churn/blob/main/Codes/Bank%20Customer%20Churn%20Prediction%20-%203_Clustering.ipynb)
We want to understand the demographics of all the customers who have churned and see if we can segment them.

We do this by fitting KMeans. We will use both Elbow Curve and Silhouette analysis to give a suggested K.

### Elbow Curve
![image](https://user-images.githubusercontent.com/63915619/221851656-f64dde32-720d-46f8-8fd9-9ddb2068d8a8.png)

### Silhouette
![image](https://user-images.githubusercontent.com/63915619/221851806-6b742b9c-cdf5-429b-b36e-c28f033c14d7.png)

All Ks have clusters with Silhouette score more than the average score of the dataset.

Only K=2 has a more evenly spread-out clusters compare to K=4.

Hence, we instantiated both k=2 and K=4 to see which give better defined clusters.

K=4 gives more interpretable clusters with 'balance', 'products_number' and 'gender' the features that help to distinguish the clusters.


### Final CLuster Model
![image](https://user-images.githubusercontent.com/63915619/221859977-913aa896-1f4b-4376-aa00-2b1cef61f4af.png)
Cluster 0 (Working Males): Male, minimum 45k balance, maximum 220k balance, average balance of 120k, mostly holds only 1 product and max 2 products

Cluster 1 (Spenders): Less than 52k balance, average balance of 1.7k, mostly holds only 1 product and max 2 products

Cluster 2 (Working Females): Female, minimum balance of 52k, maximum 230k balance, average balance of 120k, mostly holds only 1 product and max 2 products

Cluster 3 (Young Singles): Average balance 87k, 25% of the group has zero balance, 50% of the group has more than 100k balance, mostly holds at least 3 product and max 4 products
<br><br>

![image](https://user-images.githubusercontent.com/63915619/221347444-c8ce20b7-cf26-4a2a-add9-19137a81b7fa.png)

<br><br>

Note: The notebook is best viewed using Jupyter Lab.


# [Demo](https://ngjance-bank-churn-bank-customer-churn-prediction-4-demo-elqwyt.streamlit.app/)
Using Streamlit, we deploy the models to cloud so that business users can access and use the app.
<br>
![image](https://user-images.githubusercontent.com/63915619/222423703-57d497dd-246b-4f80-af2b-b548d6019ec4.png)
<br><br>

Note: This is the version 1 of the app and there may be more updates.
