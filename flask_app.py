# pip install flask requests
from flask import Flask, render_template, request, jsonify
import requests, os

app = Flask(__name__)
API = "http://localhost:8000"

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}
    if request.method == "POST":
        form_data = {
            "question": request.form["question"],
            "visualize": bool(request.form.get("visualize")),
            "provider": request.form["provider"]
        }
        payload = form_data.copy()
        r = requests.post(f"{API}/query", json=payload)
        result = r.json()
    return render_template("index.html", result=result, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True, port=5000)