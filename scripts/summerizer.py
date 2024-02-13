import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Function to summarize the blog
def summarize_blog(blog_text):
    response = client.chat.completions.create(
        model=os.environ.get("COMPLETION_MODEL"),
        messages=[
            {
                "role": "system",
                "content": """You are a helpful assistant who reads blog posts and gives a summerization for them,
                you give a short description for the items mentioned in the blog post for the first 5 items,
                you return the summary in json format like the following:
                {"summary": {"item 1": "paragraph 1", "item 2": "paragraph 2", ...}}""",
            },
            {
                "role": "user",
                "content": f"Summarize the blog delimited by the back ticks in two paragraphs:\n `{blog_text}`",
            },
        ],
    )
    return json.loads(response.choices[0].message.content)
