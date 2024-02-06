from flask import Flask, jsonify, request

from scripts.recommend_engine import get_recommendations
from scripts.summerizer import summarize_blog

app = Flask(__name__)


# API endpoint for summarizing a blog
@app.route("/summarize-blog", methods=["POST"])
def summarize():
    data = request.get_json()
    summary = summarize_blog(data.get("blog_text"))
    return jsonify(summary)


# API endpoint for getting product documentations
@app.route("/recommendations", methods=["POST"])
def recommendations():
    data = request.get_json()
    return jsonify(get_recommendations(data.get("query_string"), data.get("length", 3)))


if __name__ == "__main__":
    app.run(debug=True)
