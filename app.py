from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import sys
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Get', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)
        print(pred[0])
        return render_template('home.html', pred=pred[0])
    

if __name__ == "__main__":
    app.run(debug=True)