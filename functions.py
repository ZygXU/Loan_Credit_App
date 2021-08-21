import pandas as pd
import numpy as np
import joblib
import pycaret
import pickle
import json
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from flask import Response
from flask import session


# prepare data for prediction
def preprocessing_predict_data(df):
    # select 14 features
    # df=df.apply(lambda x: float(x))

    df['EXT_SOURCE_2 EXT_SOURCE_3'] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df['EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3'] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
    df['EXT_SOURCE_2 EXT_SOURCE_3 DAYS_BIRTH'] = df["EXT_SOURCE_2"] * df["DAYS_BIRTH"] * df["EXT_SOURCE_3"]
    df['EXT_SOURCE_2^2 EXT_SOURCE_3'] = (df["EXT_SOURCE_2"].pow(2)) * df["EXT_SOURCE_3"]
    df['EXT_SOURCE_2 EXT_SOURCE_3^2'] = (df["EXT_SOURCE_3"].pow(2)) * df["EXT_SOURCE_2"]
    df['ECONOMICAL'] = df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: np.floor(abs(x) / 100))
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    df = df[
        ['EXT_SOURCE_2 EXT_SOURCE_3', 'EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3', 'EXT_SOURCE_2 EXT_SOURCE_3 DAYS_BIRTH',
         'EXT_SOURCE_2^2 EXT_SOURCE_3', 'EXT_SOURCE_2 EXT_SOURCE_3^2',
         'EXT_SOURCE_1', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
         'ECONOMICAL', 'DAYS_EMPLOYED_PERCENT', 'DAYS_REGISTRATION', 'AMT_GOODS_PRICE']]

    return df


# compare 2 dataframe, if one is included in another
def compare(df1, df2):
    count = 0
    for i in df1.index:
        # print("-----------PRINT COMPARE--------------")
        # print(i)
        if ((df1[i] != df2[i]) & (count <= 4)):
            return False
        elif ((df1[i] != df2[i]) & (count > 4)):
            return True
        count = count + 1
    return True


def search_customer(data_1, data):
    search_result = data.apply(lambda x: compare(data_1, x), axis=1)
    print("-------SEARCH RESULT--------\n", data[search_result == True])
    return data[search_result == True]


def test_NAN(df):
    print(df.isnull())
    return df


def predict(df):
    # loaded_model = pickle.load(open('application/' + 'model_catboost.pkl', 'rb'))
    classifer = joblib.load('data/' + 'model_xgboost_recall_13v.pkl')
    result = classifer.predict(df)
    if (result == 0):
        result = "Loan Approuved (0)"
    else:
        result = "Loan rejeted (1)"

    return result


def toGraph_JSON_filter(data, customer):
    data_max = data.value_counts().iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data))
    fig.add_trace(go.Scatter(x=[customer, customer], y=[0, data_max], mode='lines', line=dict(color='red', width=5)))

    fig.update_layout(height=1500, width=2100, title_text="customer information plot")
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


def toGraph_JSON(data, customer_df_input):
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "DAYS_BIRTH", "DAYS_EMPLOYED", "REGION_RATING_CLIENT_W_CITY", "REGION_RATING_CLIENT", "DAYS_REGISTRATION",
            "AMT_GOODS_PRICE", "DAYS_LAST_PHONE_CHANGE"))  # ," "," ", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"))

    fig.add_trace(go.Histogram(x=data["DAYS_BIRTH"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[customer_df_input["DAYS_BIRTH"].values[0], customer_df_input["DAYS_BIRTH"].values[0]],
                             y=[0, 2600],
                             mode='lines', line=dict(color='red', width=5)), row=1, col=1)

    fig.add_trace(go.Histogram(x=data["DAYS_EMPLOYED"]), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=[customer_df_input["DAYS_EMPLOYED"].values[0], customer_df_input["DAYS_EMPLOYED"].values[0]],
                   y=[0, 155000],
                   mode='lines', line=dict(color='red', width=5)), row=1, col=2)

    fig.add_trace(go.Histogram(x=data["REGION_RATING_CLIENT_W_CITY"]), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[customer_df_input["REGION_RATING_CLIENT_W_CITY"].values[0],
           customer_df_input["REGION_RATING_CLIENT_W_CITY"].values[0]],
        y=[0, 230000], mode='lines', line=dict(color='red', width=5)), row=1, col=3)

    fig.add_trace(go.Histogram(x=data["REGION_RATING_CLIENT"]), row=2, col=1)
    fig.add_trace(
        go.Scatter(x=[customer_df_input["REGION_RATING_CLIENT"].values[0],
                      customer_df_input["REGION_RATING_CLIENT"].values[0]],
                   y=[0, 230000], mode='lines', line=dict(color='red', width=5)), row=2, col=1)

    fig.add_trace(go.Histogram(x=data["DAYS_REGISTRATION"]), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=[customer_df_input["DAYS_REGISTRATION"].values[0], customer_df_input["DAYS_REGISTRATION"].values[0]],
        y=[0, 2500], mode='lines', line=dict(color='red', width=5)), row=2, col=2)

    fig.add_trace(go.Histogram(x=data["AMT_GOODS_PRICE"]), row=2, col=3)
    fig.add_trace(
        go.Scatter(x=[customer_df_input["AMT_GOODS_PRICE"].values[0], customer_df_input["AMT_GOODS_PRICE"].values[0]],
                   y=[0, 40000],
                   mode='lines', line=dict(color='red', width=5)), row=2, col=3)

    fig.add_trace(go.Histogram(x=data["DAYS_LAST_PHONE_CHANGE"]), row=3, col=1)
    fig.add_trace(
        go.Scatter(x=[customer_df_input["DAYS_LAST_PHONE_CHANGE"].values[0],
                      customer_df_input["DAYS_LAST_PHONE_CHANGE"].values[0]],
                   y=[0, 40000], mode='lines', line=dict(color='red', width=5)), row=3, col=1)

    fig.update_layout(height=1500, width=2100, title_text="Multiple plot about customer information")
    graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph1JSON


# DataClean Supp/replace valeur abberante
def data_clean(df):
    df.OCCUPATION_TYPE.fillna("Undefined", inplace=True)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: np.nan if (x > 0) else x)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].apply(lambda x: df["DAYS_EMPLOYED"].median() if (np.isnan(x)) else x)

    df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].median(axis=0), inplace=True)
    df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].median(axis=0), inplace=True)
    df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].median(axis=0), inplace=True)

    print("clean ok")
    return df


def predict_group(customer_df):
    customer_df = customer_df[
        ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_REGISTRATION',
         'AMT_GOODS_PRICE', 'DAYS_LAST_PHONE_CHANGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    SScaler = pickle.load(open("data/SScaler.pkl", "rb"))
    x = SScaler.transform(customer_df)
    model = pickle.load(open("data/kmeans5.pkl", "rb"))
    print(x)
    predict_group = model.predict(x)

    return predict_group


def create_figure(customer_df, group, col):
    data_result = pd.read_csv('data/Kmeans_group_result.csv')
    # print(data_result)
    print("---------------------")

    mm0 = data_result[(data_result["result"] == group[0]) & (data_result["TARGET"] == 0)].agg(['mean', 'median'])[col]
    mm0 = mm0.rename({"mean": "0_mean", "median": "0_median"})
    mm1 = data_result[(data_result["result"] == group[0]) & (data_result["TARGET"] == 1)].agg(['mean', 'median'])[col]
    mm1 = mm1.rename({"mean": "1_mean", "median": "1_median"})
    mm = mm0.append(mm1)
    actual_val = {'actual_value': customer_df[col][0]}
    actual_val = pd.Series(actual_val)
    mm = mm.append(actual_val)
    colors = {'0_mean': 'lightgreen',
              '0_median': 'red',
              '1_mean': 'lightgreen',
              '1_median': 'red',
              'actual_value': 'blue'}
    fig = px.bar(mm, y=mm.index, x=mm, color=colors, title=col)
    fig.update_layout(autosize=False, width=1200, height=600)
    graph1JSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print("--------------ENDING JSON-------")
    return graph1JSON
