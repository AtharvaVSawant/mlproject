from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

# Home route
@app.route('/')
def index():
    return render_template('home.html')   # avoid missing index.html issue


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Safe input handling
        reading_score = request.form.get('reading_score')
        writing_score = request.form.get('writing_score')

        reading_score = float(reading_score) if reading_score else 0
        writing_score = float(writing_score) if writing_score else 0

        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('home.html', results=round(results[0], 2))

    except Exception as e:
        print("Error:", e)
        return render_template('home.html', results="Error occurred")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)