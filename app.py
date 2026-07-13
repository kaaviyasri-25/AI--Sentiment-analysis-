from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

history = []


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    emoji = ""
    text = ""

    if request.method == "POST":

        text = request.form["text"]

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)

        confidence = round(max(probability[0])*100,2)


        if prediction == "Positive":
            emoji = "😊"

        elif prediction == "Negative":
            emoji = "😞"

        else:
            emoji = "😐"


        history.append({
            "comment":text,
            "sentiment":prediction,
            "confidence":confidence
        })


    total = len(history)

    positive = len([x for x in history if x["sentiment"]=="Positive"])

    negative = len([x for x in history if x["sentiment"]=="Negative"])

    neutral = len([x for x in history if x["sentiment"]=="Neutral"])


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        emoji=emoji,
        text=text,
        history=history,
        total=total,
        positive=positive,
        negative=negative,
        neutral=neutral
    )


if __name__=="__main__":
    app.run(debug=True)
 