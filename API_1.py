import sklearn
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import dill


with open("Model\Clustering.pkl", "rb") as to_read:
    cluster = pickle.load(to_read)

with open("Model\ss.pkl", "rb") as to_read:
    ss = pickle.load(to_read)

with open("Model\enc.pkl", "rb") as to_read:
    enc = pickle.load(to_read)

with open("Model\ohe.pkl", "rb") as to_read:
    ohe = pickle.load(to_read)

def get_df():
    df=pd.read_csv('../Data/Bank_Customer_Churn_Data.csv')
    return df

def clustering_transform(customer_id):
    df = get_df()

    df[['country']] = enc.transform(df[['country']])
    df['gender'] = ohe.transform(df[['gender']].to_numpy().reshape(-1, 1)).toarray()

    filtered_cust = df[df['customer_id'] == customer_id]

    X_final = filtered_cust[['balance', 'products_number', 'gender']]
    X_ss_final = ss.transform(X_final)

    print(X_ss_final)

    return X_ss_final
