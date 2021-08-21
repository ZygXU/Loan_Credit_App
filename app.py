from flask import Flask, render_template, request, session
from functions import *
import os
import sys
import logging

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/customer")
def customer():
    customer_df_input = session.get('customer_df_input')
    customer_df_input = pd.read_json(customer_df_input, orient='records')
    data = pd.read_csv('data/application_train.csv')
    customer_df_find = search_customer(customer_df_input.iloc[0], data)
    if customer_df_find.empty:
        print('DataFrame is empty!')
        customer_df_find = customer_df_input
    session['customer_df_find'] = customer_df_find.to_json()
    list_customer_data = customer_df_find.values.tolist()[0]
    list_customer_columns = customer_df_find.columns.values.tolist()
    print(list_customer_data)
    print(list_customer_columns)

    return render_template('customer.html', list_customer_data=list_customer_data,
                           list_customer_columns=list_customer_columns, len=len(list_customer_columns))


@app.route("/customer_filter", methods=["POST"])
def customer_filter():
    customer_df_find = session.get('customer_df_find')
    customer_df_find = pd.read_json(customer_df_find, orient='records')
    filter_col_select = request.form.get("filter_col_select")
    print("filter_col_select:", filter_col_select)
    value = customer_df_find[filter_col_select][0]
    data = pd.read_csv('data/application_train.csv')
    graphJSON = toGraph_JSON_filter(data[filter_col_select], value)
    return render_template('customer_filter.html', filter_col_select=filter_col_select, value=value,
                           graphJSON=graphJSON)


@app.route("/customer_groupe")
def customer_groupe():
    # data_result = pd.read_csv('data/Kmeans_group_result.csv')

    customer_df_input = session.get('customer_df_input')
    customer_df_input = pd.read_json(customer_df_input, orient='records')
    group = predict_group(customer_df_input)
    predict_result = session.get('result')

    gJSON1 = create_figure(customer_df_input, group, "DAYS_BIRTH")
    gJSON2 = create_figure(customer_df_input, group, "DAYS_EMPLOYED")
    gJSON3 = create_figure(customer_df_input, group, "DAYS_REGISTRATION")
    gJSON4 = create_figure(customer_df_input, group, "AMT_GOODS_PRICE")
    gJSON5 = create_figure(customer_df_input, group, "DAYS_LAST_PHONE_CHANGE")
    gJSON6 = create_figure(customer_df_input, group, "REGION_RATING_CLIENT_W_CITY")
    gJSON7 = create_figure(customer_df_input, group, "REGION_RATING_CLIENT")
    gJSON8 = create_figure(customer_df_input, group, "EXT_SOURCE_1")
    gJSON9 = create_figure(customer_df_input, group, "EXT_SOURCE_2")
    gJSON10 = create_figure(customer_df_input, group, "EXT_SOURCE_3")

    return render_template('customer_groupe.html', group=group, predict_result=predict_result, gJSON1=gJSON1,
                           gJSON2=gJSON2, gJSON3=gJSON3, gJSON4=gJSON4, gJSON5=gJSON5, gJSON6=gJSON6, gJSON7=gJSON7,
                           gJSON8=gJSON8, gJSON9=gJSON9, gJSON10=gJSON10)


@app.route("/result", methods=["POST"])
def result():
    DAYS_BIRTH = float(request.form.get("DAYS_BIRTH"))
    DAYS_EMPLOYED = float(request.form.get("DAYS_EMPLOYED"))
    REGION_RATING_CLIENT_W_CITY = float(request.form.get("REGION_RATING_CLIENT_W_CITY"))
    REGION_RATING_CLIENT = float(request.form.get("REGION_RATING_CLIENT"))
    DAYS_REGISTRATION = float(request.form.get("DAYS_REGISTRATION"))
    AMT_GOODS_PRICE = float(request.form.get("AMT_GOODS_PRICE"))
    DAYS_LAST_PHONE_CHANGE = float(request.form.get("DAYS_LAST_PHONE_CHANGE"))
    EXT_SOURCE_1 = float(request.form.get("EXT_SOURCE_1"))
    EXT_SOURCE_2 = float(request.form.get("EXT_SOURCE_2"))
    EXT_SOURCE_3 = float(request.form.get("EXT_SOURCE_3"))
    input_data = [{'DAYS_BIRTH': DAYS_BIRTH, 'DAYS_EMPLOYED': DAYS_EMPLOYED,
                   'REGION_RATING_CLIENT_W_CITY': REGION_RATING_CLIENT_W_CITY,
                   'REGION_RATING_CLIENT': REGION_RATING_CLIENT, 'DAYS_REGISTRATION': DAYS_REGISTRATION,
                   'AMT_GOODS_PRICE': AMT_GOODS_PRICE, 'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
                   'EXT_SOURCE_1': EXT_SOURCE_1, 'EXT_SOURCE_2': EXT_SOURCE_2, 'EXT_SOURCE_3': EXT_SOURCE_3}]
    df = pd.DataFrame(input_data)
    session['customer_df_input'] = df.to_json()

    data = pd.read_csv('data/application_train.csv')

    # preprocessing+predict result
    df_preprocess = preprocessing_predict_data(df)
    result = predict(df_preprocess)
    print('result predict:', result)
    session['result'] = result

    # session['customer_df'] = customer_df.to_json()
    print(df)
    graph1JSON = toGraph_JSON(data, df)
    return render_template('result_graph.html', graph1JSON=graph1JSON, result=result)


if __name__ == "__main__":

    app.config['SESSION_TYPE'] = 'filesystem'

    # session.init_app(app)
    app.debug = True
    app.run()
