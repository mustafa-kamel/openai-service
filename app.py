import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Function to summarize the blog
def summarize_blog(blog_text):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Summarize the following blog in two paragraphs:\n",
            },
            {
                "role": "user",
                "content": blog_text,
            },
        ],
    )
    return response.choices[0].message.content


# API endpoint for summarizing blog
@app.route("/summarize-blog", methods=["POST"])
def summarize():
    data = request.get_json()
    summary = summarize_blog(data["blog_text"])
    print(summary)
    return jsonify({"summary": summary})


if __name__ == "__main__":
    app.run(debug=True)
