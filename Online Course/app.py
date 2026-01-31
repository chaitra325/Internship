# app.py
from flask import Flask, render_template, request, redirect, url_for
from preprocessing import predict_success
from openai_advisor import generate_course_suggestions
import os
from dotenv import load_dotenv

# load .env
load_dotenv()

# Use API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")


app = Flask(__name__)

# Page 1: input form 
@app.route("/", methods=["GET"])
def input_page():
    return render_template("input.html")

# Handle form: ML + GenAI, then redirect
@app.route("/predict", methods=["POST"])
def predict():
    # 1) Read form values
    price = float(request.form.get("price", 0))
    reviews = int(request.form.get("reviews", 0))
    rating = float(request.form.get("rating", 0))
    duration = float(request.form.get("duration", 0))
    lecture_numbers = int(request.form.get("lecture_numbers", 0))

    course_title = request.form.get("title") or "Custom Course"
    course_category = request.form.get("category")
    difficulty = request.form.get("difficulty")

    raw_dict = {
        "title": course_title,
        "category": course_category,
        "difficulty": difficulty,
        "price": price,
        "reviews": reviews,
        "rating": rating,
        "duration": duration,
        "lecture_numbers": lecture_numbers,
        # instructor proxies inferred from this course
        "instr_total_reviews": reviews,
        "instr_mean_rating": rating if rating > 0 else 4.3,
        "instr_course_count": 1,
    }

    # 2) ML prediction
    label, proba_val = predict_success(raw_dict)
    prediction = "High success" if label == 1 else "Low success"
    proba_str = f"{proba_val:.2f}"

    # 3) OpenAI suggestions
    suggestions = generate_course_suggestions(raw_dict, label, proba_val)

    # 4) Redirect to /result, passing everything via query string
    return redirect(url_for(
        "result_page",
        title=course_title,
        category=course_category,
        prediction=prediction,
        proba=proba_str,
        suggestions=suggestions
    ))

#  Page 2: show result + AI text 
@app.route("/result", methods=["GET"])
def result_page():
    course_title = request.args.get("title")
    course_category = request.args.get("category")
    prediction = request.args.get("prediction")
    proba = request.args.get("proba")
    suggestions = request.args.get("suggestions")

    return render_template(
        "result.html",
        course_title=course_title,
        course_category=course_category,
        prediction=prediction,
        proba=proba,
        suggestions=suggestions,
    )

if __name__ == "__main__":
    app.run(debug=True)
