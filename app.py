from flask import Flask, jsonify, request

from scripts.summerizer import summarize_blog

app = Flask(__name__)


# API endpoint for summarizing blog
@app.route("/summarize-blog", methods=["POST"])
def summarize():
    data = request.get_json()
    summary = summarize_blog(data["blog_text"])
    return jsonify(summary)


if __name__ == "__main__":
    app.run(debug=True)
