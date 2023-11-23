from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contactmail', methods=['POST'])
def contactmail():
    # Process the form data here
    name = request.form.get('name')
    email = request.form.get('email')
    subject = request.form.get('subject')
    message = request.form.get('message')

    # Add your logic to handle the form data (e.g., send an email, store in a database)

    # For now, print the form data
    print(f"Name: {name}, Email: {email}, Subject: {subject}, Message: {message}")

    return "Form submitted successfully!"

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/fertilizer')
def fertilizer():
    return render_template('fertilizer.html')

@app.route('/wheat')
def wheat():
    return render_template('wheat.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from the form
    nitrogen = float(request.form.get('Nitrogen'))
    potassium = float(request.form.get('Potassium'))
    phosphorous = float(request.form.get('Phosphorous'))

    # Load the trained model and scaler
    with open('knn_model.pkl', 'rb') as model_file:
        knn_model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Use the model to make a prediction based on the input data
    numerical_result = knn_model.predict(scaler.transform([[nitrogen, potassium, phosphorous]]))

    # Map the numerical result to categorical valuef
    category_mapping = {
        0: "DAP",
        1: "Fourteen-Thirty Five-Fourteen",
        2: "Seventeen-Seventeen-Seventeen",
        3: "Ten-Twenty Six-Twenty Six",
        4: "TWENTY EIGHT-TWENTY EIGHT",
        5: "TWENTY-TWENTY",
        6: "UREA"
    }

    categorical_result = category_mapping.get(numerical_result[0], "Unknown")

    # Additional information for each fertilizer category
    fertilizer_info = {
        "UREA": {
            "description": "Urea is the most important nitrogenous fertilizer in the market, with the highest Nitrogen content (about 46 percent). It is a white crystalline organic chemical compound. Urea is neutral in pH and can adapt to almost all kinds of soils.",
            "crops": ["Wheat", "Sugar beet", "Perennial grasses", "Grain corn", "Silage corn", "Oats", "Rye", "Barley", "Potatoes", "Sunflower", "Soybeans", "Rapeseed"],
            "how_to_use": ["Urea should be applied at the time of sowing. It should not come in contact with the seeds. It also can be applied as a top dressing.",
                           "Since Urea is highly Nitrogen-concentrated, it should be used in combination with earth or sand before its application.",
                           "Urea should not be applied when the soil contains free water or likely to remain wet for three or four days after application."],
            "price": "Rs 5,360 per tonne i.e. Rs. 268 per 50 kg bag",
            "image_path": "static/img/fertilizer/urea.jpg"
        },
        "DAP": {
            "description": "Dap is known as Diammonium phosphate. It can be applied in autumn for tilling and in spring during sowing, as well as for pre-sowing cultivation. Dissolving in soil, it provides temporary alkalization of pH of the soil solution around the fertilizer granule, thus stimulating better uptake of phosphorus from the fertilizers on acid soils.",
            "crops": ["Wheat", "Sugar beet", "Perennial grasses", "Grain corn", "Silage corn", "Oats", "Rye", "Barley", "Potatoes", "Sunflower", "Soybeans", "Rapeseed"],
            "how_to_use": ["Apply 5 to 10 granules for 6 to 12 inch planter",
                           "Advised to till/ mix the soil after 1 to 2 days of using dap",
                           "Use not more than once in 15 days.",
                           "Apply either during pre sowing cultivation, tilling or during sowing of crops"],
            "price": "Rs. 1125 /50 kg bag",
            "image_path": "static/img/fertilizer/Dap.png"
        },
        "Fourteen-Thirty Five-Fourteen": {
            "description": "Best for potted plants, hydroponic, flower plants, fruit plants, vegetables, and ever-green plants. The numbers 14-35-14 represent the N-P-K (Nitrogen-Phosphorus-Potassium) ratio in a fertilizer. This type of fertilizer is suitable for crops that require an extra boost during the early stages of growth or during the reproductive phase.",
            "crops": ["Wheat", "Rice", "Cotton", "Groundnut", "Chillies", "Soya bean", "Potato"],
            "how_to_use": ["You can apply the fertilizer during wheat sowing or planting. Broadcast the fertilizer evenly across the soil before seeding or during seeding.",
                           "As the wheat plants grow, you can apply a topdressing of fertilizer when the plants are in the early stages of development"],
            "price": "Rs. 1096 /50 kg bag",
            "image_path": "static/img/fertilizer/14-35-14.jpg"
        },
        "Ten-Twenty Six-Twenty Six": {
            "description": "Fertilizers with a higher phosphorus and potassium content, such as 10-26-26, are often used to promote flowering, fruiting, and root development. Suitable for crops that require additional support during the reproductive phase. Commonly used in fruiting plants, vegetables, and crops that benefit from increased phosphorus and potassium.",
            "crops": ["Tomatoes", "Peppers", "Eggplants", "Carrots", "Potatoes", "Roses", "Apple", "Pear", "Peach"],
            "how_to_use": ["Incorporate the fertilizer into the soil before planting or sowing seeds.",
                           "For established plants, you can apply the fertilizer around the base of each plant.",
                           "Apply the fertilizer as a topdressing during the growing season.",
                           "Ensure even distribution around the root zone of the plants.",
                           "Avoid direct contact with plant foliage to prevent burning.",
                           "Water the area thoroughly after applying the fertilizer to help nutrients move into the soil."],
            "price": "Rs. 1660 /50 kg",
            "image_path": "static/img/fertilizer/10-26-26.jpg"
        },
        "TWENTY EIGHT-TWENTY EIGHT": {
            "description": "This is the highest Nitrogen containing complex fertilizer with 28%. It provides immediate nutrition to the crop during the peak growth period. It is virtually free from detrimental elements like Chloride and Sodium. It can be sprayed on all types of vegetables, fruit crops, cereals, and pulses for boosting the crop growth thus getting better yield and quality.",
            "crops": ["Paddy", "Cotton", "Chillies", "Sugarcane", "Vegetables"],
            "how_to_use": ["0.5 – 1.0% concentration (5-10 gm / lit of water) to be used as foliar spray during vegetative and flowering stages.",
                           "Apply 1 kg/acre",
                           "Method of application is: Spray or soil application",
                           "Duration of effect is 15 days"],
            "price": "Rs. 1120 /50 kg",
            "image_path": "static/img/fertilizer/28-28.png"
        },
        "Seventeen-Seventeen-Seventeen": {
            "description": "17:17:17 contains the most important primary nutrients Nitrogen, Phosphorous, and Potash in equal proportion. Suitable for all crops both for initial application and top dressing. Granules are stronger, harder, and of uniform size which facilitates easy application.",
            "crops": ["Corn", "Wheat", "Barley", "Oat", "Sorghum", "Tomatoes", "Peppers", "Cucumbers", "Soybeans", "Peas", "Potatoes", "Sweet potatoes"],
            "how_to_use": ["Mix 2 - 4 gm of NPK per Liter of water for all crops, flower, vegetables, plantation, indoor & Outdoor Plants.",
                           "Newly planted trees and shrubs will get the nutrients necessary to sustain healthy growth in their development's early and fast-growing phases by applying 17-17-17 fertilizer to the planting hole."],
            "price": "Rs. 1240 /50 kg",
            "image_path": "static/img/fertilizer/17-17-17.jpg"
        },
        "TWENTY-TWENTY": {
            "description": "Suitable for a variety of crops, including those with moderate potassium needs. Commonly used during the early stages of plant growth, such as planting or transplanting. Crops that benefit from a balanced nitrogen-phosphorus fertilizer without immediate potassium supplementation.",
            "crops": ["Wheat", "Barley", "Oats", "Lettuce", "Spinach", "Kale", "Cabbage", "Broccoli", "Cauliflower", "Carrots", "Radishes"],
            "how_to_use": ["Apply the 20-20-0 fertilizer during the appropriate growth stages for your crops.",
                           "Common application times include at planting, during transplanting, or early in the growing season.",
                           "Avoid concentrated application in one area, as this may lead to uneven growth."],
            "price": "Rs. 525 / 50 kg",
            "image_path": "static/img/fertilizer/20-20.jpg"
        }
    }

    # Render the prediction result and additional information in your HTML template
    return render_template('fertilizer.html', result=categorical_result, fertilizer_info=fertilizer_info.get(categorical_result, None))


if __name__ == '__main__':
    app.run(debug=True)
