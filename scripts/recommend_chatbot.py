import json
import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
RECOMMEND_COMPLETION_MODEL = os.environ.get("RECOMMEND_COMPLETION_MODEL")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
dataset_path = os.environ.get("DATASET_PATH")
embedding_cache_path = os.environ.get("EMBEDDING_CACHE_PATH")
df = pd.read_csv(dataset_path)

user_messages = []


def get_completion(text: str, role: str = "user", model=RECOMMEND_COMPLETION_MODEL, **kwargs) -> dict:
    # replace newlines, which can negatively affect performance.
    user_messages.append({"role": role, "content": text})
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are a helpful chatbot who helps customers make orders,
                    follow the following steps to help them order items,
                1- you detect the language of the user message between the back ticks and reply to him in same language
                2- if the customer message language changed you also change your reply language to the new language
                3- you return a helpful response in JSON format that should contain your reply and recommended items
                    you return the response in json format like the following:
                    {"reply": "YOUR REPLY MESSAGE",
                    "recommend": true or false,
                    "products": "COMMA-SEPARATED items STRING",
                    "cart": {"ITEM 1": "QUANTITY", "ITEM 2": "QUANTITY", ...}}
                4- you reply briefly in the `reply` field and keep your tone helpful and friendly
                5- each time the customer sends a message extract any product names in his message and translate them
                into Arabic/English, then set the `products` field as a comma-separated string contianing original
                products names and their translations, and set the `recommend` field to True,
                otherewise `products` field should be an empty string, and `recommend` flag is False,
                6- the `products` field only contians data from the user message and not from the system recommendations
                7- system recommendations should be added to the `reply` field only, not to the `products` field
                8- if you didn't understand the question, ask the customer for calrification
                9- you also add ordered items (after confirming the items and quantities) to the response `cart` field
                    along with the requested quantities in the following format:
                    "cart": {"FULL RECOMMENDED ITEM 1 NAME": QUANTITY, "ITEM 2 FULL NAME": QUANTITY, ...}
                10- when the user requests to order an item you should add the full item name from the recommendations
                to the cart and its quantity, you only add items to cart after confirmation with the customer about
                which items will you add to cart and how much of it
                11- when you get system product recommendations you should suggest the 5 most relevant items in the
                `reply` field
                12- you must follow the customer orders until you satisfy all his requirements
                13- when you are done with one item and adding it to the cart after confirmation, keep recommending
                    items based to the customer, or continue with other items he asked you about
                """,
            }
        ]
        + user_messages,
        **kwargs,
    )
    return json.loads(response.choices[0].message.content)


def get_embedding(text: str, model=EMBEDDING_MODEL, **kwargs) -> List[float]:
    """Send a request to OpenAI embedding endpoint to retrieve the embedding for the specified string
    unsing the selected model
    """

    # replace newlines, which can negatively affect performance.
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


if "embedding" not in df.columns:
    df["name"] = df.apply(lambda x: x["name_ar"] + " - " + x["name_en"], axis=1)
    df["embedding"] = df.name.apply(lambda x: get_embedding(x))
    df.to_csv(dataset_path, index=False)
else:
    df["embedding"] = df["embedding"].apply(eval).apply(np.array)


def chatbot_interaction(user_text):
    # Generate chat completions for the cleaned message
    user_text = user_text.replace("\n", " ")
    response = get_completion(f"Reply to the customer message delimited by the back ticks:\n `{user_text}`")

    if response.get("recommend") is True:
        # Generate embeddings for the cleaned message
        user_embedding = get_embedding(response.get("products", user_text))
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, user_embedding))
        # Sort products by similarity
        items = np.argsort(df["similarity"])[:-15:-1]

        items["display_name"] = df.loc[items].apply(
            lambda x: x["name_ar"] + " " + x["name_en"] + " $" + str(x["price"]), axis=1
        )
        items = items["display_name"].tolist()
        response = get_completion(
            "Reply to the customer using the following recommended items list [" + ", ".join(items) + "]",
            role="system",
        )
    print(response)
    return response


def main():
    print("Hello, How can I help you!")
    while True:
        user_text = input("You: ")
        if user_text.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!", user_messages)
            break
        else:
            response = chatbot_interaction(user_text)
            user_messages.append({"role": "user", "content": f"{response}"})
            print("Assistant:", response.get("reply", response))


if __name__ == "__main__":
    main()
