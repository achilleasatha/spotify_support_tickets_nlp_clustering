from flask import Flask, request, jsonify, render_template
import pickle
import os
import json
import pandas as pd
import datetime


app = Flask(__name__)
model = pickle.load(open(os.getcwd() + r'/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req_dict = request.form.to_dict(flat=False)
    # data = format_data(data)
    prediction = model.predict(pd.DataFrame.from_dict(req_dict))
    output = prediction[0]
    return render_template('index.html', prediction_text='Ticket has been classified '
                                                         'as: {}'.format(output))


@app.route('/results', methods=['POST'])
def results():
    data = request.get_json(force=True)
    data = format_data(data)
    prediction = model.predict(pd.read_json(json.dumps(data), lines=True))

    output = prediction[0]
    return jsonify(output)


def format_data(data):
    timestamp = datetime.datetime.now()
    # todo generate any additional temporal features and append to data object
    return data


def get_cluster_name(x):
    # todo not sure if this should live here or as a static method in our data parser
    cluster_dict = {'cluster_id': 'cluster_name'}
    return cluster_dict[x]

if __name__ == "__main__":
    app.run(debug=False)
