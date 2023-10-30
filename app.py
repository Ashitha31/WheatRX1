from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    nitrogen = float(request.form.get('Nitrogen'))
    potassium = float(request.form.get('Potassium'))
    phosphorous = float(request.form.get('Phosphorous'))

    # Load the model (classifier) from the saved file
    with open('classifier1.pkl', 'rb') as model_file:
        classifier = pickle.load(model_file)

    # Define the category mapping
    category_mapping = {
        0: "TEN-TWENTY SIX-TWENTY SIX",
        1: "Fourteen-Thirty Five-Fourteen",
        2: "Seventeen-Seventeen-Seventeen",
        3: "TWENTY-TWENTY",
        4: "TWENTY EIGHT-TWENTY EIGHT",
        5: "DAP",
        6: "UREA"
    }

    # Use the model to make a prediction based on the input data
    numerical_result = classifier.predict([[nitrogen, potassium, phosphorous]])

    # Map the numerical result to categorical value
    categorical_result = category_mapping.get(numerical_result[0], "Unknown")

    # Render the prediction result in your HTML template
    return render_template('fertilizer.html', result=categorical_result)

if __name__ == '__main__':
    app.run(debug=True, port=8080)
