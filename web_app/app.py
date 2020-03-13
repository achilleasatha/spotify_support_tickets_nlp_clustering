from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd


app = Flask(__name__)
model = pickle.load(open(os.getcwd() + r'/../model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req_dict = request.form.to_dict(flat=False)
    prediction = model.predict(pd.DataFrame.from_dict(req_dict))
    topic_map = {0: 'General', 1: 'Update related issue', 2: 'Content related query', 3: 'Technical issue',
                 4: 'Subscription and account related issue'}
    output = topic_map[prediction[0]]
    return render_template('index.html', prediction_text='Ticket has been classified '
                                                         'as: {}'.format(output))


@app.route('/results', methods=['POST'])
def results():
    req_dict = request.form.to_dict(flat=False)
    prediction = model.predict(pd.DataFrame.from_dict(req_dict))
    topic_map = {0: 'General', 1: 'Update related issue', 2: 'Content related query', 3: 'Technical issue',
                 4: 'Subscription and account related issue'}
    output = topic_map[prediction[0]]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)
