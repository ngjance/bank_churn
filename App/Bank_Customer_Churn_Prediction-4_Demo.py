import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle

#import the models
from API import get_df, clustering_transform, predict_transform, cluster_plot

with open("Model\Clustering.pkl", "rb") as to_read:
    cluster = pickle.load(to_read)

with open("Model\lightgbm.pkl", "rb") as to_read:
    model_clf = pickle.load(to_read)

#set the page layout
st.set_page_config(page_title="Bank_Customer_Churn_Prediction-4_Demo.py",
                   layout="wide"
                   )

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 3rem;
                    padding-left: 1rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.markdown("""
           <style>
           .title_style {
               font-size:45px;
               text-align: center;
               font-weight: bold;               
           }
           </style>
           """, unsafe_allow_html=True)

st.markdown('<p class="title_style">Client Master Application</p>',
                unsafe_allow_html=True)

df = get_df()

#sidebar
st.sidebar.header("Enter customer ID here")

cust_id = st.sidebar.selectbox(
            label="",
            label_visibility='collapsed',
            options=df['customer_id'].unique()
        )
st.sidebar.markdown('<br>',
                unsafe_allow_html=True)
st.sidebar.header("Is Client Likely to Churn?")

predict_churn = predict_transform(cust_id)

churn = model_clf.predict(predict_churn)

st.sidebar.markdown("""
           <style>
           .sidebar-font {
               font-size:20px             
           }
           </style>
           """, unsafe_allow_html=True)

if churn == 1:

    st.sidebar.markdown('<p class="sidebar-font">Yes</p>',
                unsafe_allow_html=True)

else:
    st.sidebar.markdown('<p class="sidebar-font">No</p>',
                        unsafe_allow_html=True)


st.markdown("---")

#main
dataset = st.container()
model = st.container()

with dataset:
    st.markdown("""
           <style>
           .big-font {
               font-size:38px;
           }
           </style>
           """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Client Profile</p>',
                unsafe_allow_html=True)

    colorscale = [[0, '#034694'], [.5, '#034694'], [1, '#F0F8FF']]


    data = df[df['customer_id'] == cust_id].drop('churn', axis=1
                                                 )

    fig = ff.create_table(data, colorscale=colorscale)
    fig.layout.width = 1400

    # Make text size larger
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 15
        fig.layout.annotations[i].align = "center"

    st.write(fig)
    st.markdown('---')

with model:
    st.markdown("""
                       <style>
                       .mid-font {
                           font-size:30px;
                       }
                       </style>
                       """, unsafe_allow_html=True)

    st.markdown("""
                           <style>
                           .small-font {
                               font-size:25px;
                           }
                           </style>
                           """, unsafe_allow_html=True)

    if churn == 1:

        st.markdown('<p class="mid-font">If client likely to churn, which cluster does this client likely belong to?</p>',
                    unsafe_allow_html=True)

        # st.markdown('<br>',
        #             unsafe_allow_html=True)

        predict_cl = clustering_transform(cust_id)

        c = cluster.predict(predict_cl)

        st.markdown("""
        <style>
        div[data-testid="metric-container"] {
               background-color: #F0F8FF;
           border: 1px solid #C3B1E1;
           padding: 5% 5% 5% 10%;
           border-radius: 5px;
           color: #51414F;
           font-weight: bold;
           overflow-wrap: break-word;
        }

        /* breakline for metric text         */
        div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
           overflow-wrap: break-word;
           white-space: break-spaces;
           color: black;
        }
        </style>
        """
                    , unsafe_allow_html=True)

        if c == 0:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Cluster", "0 (Pro. Males)")
            col2.metric("Gender", "Male")
            col3.metric("Balance", "$45k to $220k")
            col4.metric("Average Balance", "$120k")
            col5.metric("No. of Products", "1 to 2")

            st.markdown('---')

            st.markdown(
                '<p class="mid-font">Recommendations:'
                '<br>''</p>',
                unsafe_allow_html=True)

            st.markdown(
                '<p class="small-font">- Retain Client'
                '<br>'
                '- Entice client to sign up for new products'
                '<br>'
                '- Promotions and offers relating to electronics and gadgets'
                '</p>',
                unsafe_allow_html=True)

        elif c == 1:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cluster", "1 (Spenders)")
            col2.metric("Max. Balance", "$52k")
            col3.metric("Average Balance", "$1.7k")
            col4.metric("No. of Products", "1 to 2")

            st.markdown('---')

            st.markdown(
                '<p class="mid-font">Recommendations:'
                '<br>''</p>',
                unsafe_allow_html=True)

            st.markdown(
                '<p class="small-font">- Let client churn</p>',
                unsafe_allow_html=True)

        elif c == 2:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Cluster", "2 (Pro. Females)")
            col2.metric("Gender", "Female")
            col3.metric("Balance", "$52k to $230k")
            col4.metric("Average Balance", "$120k")
            col5.metric("No. of Products", "1 to 2")

            st.markdown('---')

            st.markdown(
                '<p class="mid-font">Recommendations:'
                '<br>''</p>',
                unsafe_allow_html=True)

            st.markdown(
                '<p class="small-font">- Retain Client'
                '<br>'
                '- Entice client to sign up for new products'
                '<br>'
                '- Promotions and offers relating to beauty and shopping'
                '</p>',
                unsafe_allow_html=True)

        elif c == 3:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Cluster", "3 (Young Singles)")
            col2.metric("Balance", "$0 to $250k")
            col3.metric("Average Balance", "$87k")
            col4.metric("No. of Products", "3 to 4")

            st.markdown('---')

            st.markdown(
                '<p class="mid-font">Recommendations:'
                '<br>''</p>',
                unsafe_allow_html=True)

            st.markdown(
                '<p class="small-font">- Retain Client'
                '<br>'
                '- Introduce them investment and savings plans'
                '<br>'
                '- Promotions and offers relating to accumulating more cashbacks'
                '</p>',
                unsafe_allow_html=True)

    else:
        st.markdown('<p class="mid-font">Client is a loyal customer</p>',
                    unsafe_allow_html=True)

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
