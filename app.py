from flask import Flask, render_template, request, jsonify
from chat import get_response
from pymongo import MongoClient

# MONGO_URL = 'mongodb+srv://adi_285s:Aditya285@vishwa-bot.7syju.mongodb.net/'
# conn = MongoClient(MONGO_URL)

app = Flask(__name__)

@app.get("/")
def index_get():
    return render_template("index.html")

@app.post("/predict")
def predict():
    data = request.json
    text = data.get("message")
    print(text)
    response = get_response(text)
    print(response)

    message = {"answer": response}

    # data_dict = dict(query = text, response = response)
    # conn.CHatdata.Questions.insert_one(data_dict)

    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)
