
from flask import Flask, render_template, request
import joblib
import requests

app = Flask(__name__)
model = joblib.load("model/sentiment_model.pkl")

def analyze_comment(comment):
    prediction = model.predict([comment])[0]
    return prediction

def get_comments_from_facebook(post_id, access_token):
    url = f"https://graph.facebook.com/v18.0/{post_id}/comments?access_token={access_token}&limit=100"
    try:
        response = requests.get(url)
        data = response.json()
        comments = [item["message"] for item in data.get("data", []) if "message" in item]
        return comments
    except Exception as e:
        print("Lỗi:", e)
        return []

@app.route("/", methods=["GET", "POST"])
def index():
    comments, positives, negatives = [], [], []
    chart_data = {"positive": 0, "negative": 0}
    if request.method == "POST":
        post_id = request.form["post_id"]
        access_token = request.form["access_token"]
        comments = get_comments_from_facebook(post_id, access_token)
        for cmt in comments:
            if analyze_comment(cmt) == 1:
                positives.append(cmt)
            else:
                negatives.append(cmt)
        total = len(positives) + len(negatives)
        if total > 0:
            chart_data["positive"] = int(len(positives) / total * 100)
            chart_data["negative"] = 100 - chart_data["positive"]
    return render_template("index.html", positives=positives, negatives=negatives, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=False)

