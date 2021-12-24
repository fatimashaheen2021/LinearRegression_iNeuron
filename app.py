import pandas as pd

from flask import Flask, render_template, request, Response
from flask_cors import CORS,cross_origin

import pickle


model = pickle.load(open('lin_reg.sav', 'rb'))

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])  # To render Homepage
@cross_origin()
def home_page():
    return render_template('index.html')


@app.route('/result', methods=['POST'])  # To render Result page
@cross_origin()
def result():
    if request.method == 'POST':
        try:
            input_dict = request.form.to_dict()
            print(input_dict)
            input_features = {k.strip(): [float(v)] for k, v in input_dict.items() if k != 'CHAS'}
            input_features['CHAS'] = [1] if input_dict['CHAS'] == 'Yes' else [0] # CHAS is a binary feature
            print(input_features)
            data = pd.DataFrame(input_features)

            pred = model.predict(data)[0]
            result = pred * 1000
            print(f'Result is: {result}')

            return render_template('result.html', result=result)
            
        except Exception as e:
            print('The Exception message is: \n',e)
            return 'Something went wrong'
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

