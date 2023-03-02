import sklearn
import pandas as pd
import pickle

import streamlit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

with open("Model/Clustering.pkl", "rb") as to_read:
    cluster = pickle.load(to_read)

with open("Model/ss.pkl", "rb") as to_read:
    ss = pickle.load(to_read)

with open("Model/enc.pkl", "rb") as to_read:
    enc = pickle.load(to_read)

with open("Model/ohe.pkl", "rb") as to_read:
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

    return X_ss_final

def cluster_plot(i):
    df = get_df()

    df[['country']] = enc.transform(df[['country']])
    df['gender'] = ohe.transform(df[['gender']].to_numpy().reshape(-1, 1)).toarray()

    X_final = df.loc[lambda df: df['churn'] == 1][['balance', 'products_number', 'gender']]

    X_ss_final = ss.transform(X_final)

    cluster.fit(X_ss_final)

    X_final['cluster'] = cluster.labels_

    X_final = pd.concat([df['customer_id'], X_final], axis=1)
    X_final_i = X_final[X_final['cluster']==i]

    return X_final_i

with open("Model/enc_clf.pkl", "rb") as to_read:
    enc_clf = pickle.load(to_read)

with open("Model/ohe_clf.pkl", "rb") as to_read:
    ohe_clf = pickle.load(to_read)

with open("Model/ss_clf.pkl", "rb") as to_read:
    ss_clf = pickle.load(to_read)

with open("Model/lightgbm.pkl", "rb") as to_read:
    model_clf = pickle.load(to_read)

def predict_transform(customer_id):
    df = get_df()

    df[['country']] = enc_clf.transform(df[['country']])
    df['gender'] = ohe_clf.transform(df[['gender']].to_numpy().reshape(-1, 1)).toarray()

    df['country*balance'] = df['country'] * df['balance']

    filtered_cust = df[df['customer_id'] == customer_id]

    X_train1 = filtered_cust[['estimated_salary', 'balance', 'age','credit_score', 'country*balance','products_number','gender']]
    X_train1_s = ss_clf.transform(X_train1)

    return X_train1_s