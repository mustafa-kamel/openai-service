# Flask Summarizer

Flask Summarizer is a Python application that utilizes Flask and OpenAI to summarize lengthy blogs into concise paragraphs with ease.

## Features

- Summarize lengthy blog posts into brief, insightful paragraphs.
- Integration with OpenAI for advanced natural language processing capabilities.
- Simple HTTP request interface for easy interaction.
- Customizable summarization parameters.
- Robust error handling and response validation.
- Lightweight and efficient Flask framework for deployment and scalability.

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/flask-summarizer.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Set up your OpenAI API key by replacing `'YOUR_API_KEY'` in the code with your actual key.

## Usage

To use the Flask Summarizer API, follow these steps:

1. Run the Flask application:

```
python app.py
```

2. Send a POST request to the `/summarize-blog` endpoint with a JSON payload containing the blog text.

Example:

```
curl -X POST -H "Content-Type: application/json" -d '{"blog_text": "Replace this with your blog text."}' http://localhost:5000/summarize-blog
```
