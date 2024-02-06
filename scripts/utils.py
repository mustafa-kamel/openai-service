import os
import pickle
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy import spatial

load_dotenv()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
dataset_path = os.environ.get("DATASET_PATH")
embedding_cache_path = os.environ.get("EMBEDDING_CACHE_PATH")
df = pd.read_csv(dataset_path)

# Load the cache if it exists, and save a copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


def get_embedding(text: str, model=EMBEDDING_MODEL, **kwargs) -> List[float]:
    """Send a request to OpenAI embedding endpoint to retrieve the embedding for the specified string
    unsing the selected model
    """

    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model, **kwargs)
    return response.data[0].embedding


def embedding_from_string(string: str, model: str = EMBEDDING_MODEL, embedding_cache=embedding_cache) -> list:
    """Return embedding of given string, using a cache to avoid recomputing.
    Retrieves embeddings from the cache if present, and otherwise request via the API and save it
    """

    string = str(string)
    if string not in embedding_cache.keys():
        embedding_cache[string] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[string]


def distances_from_embeddings(
    query_embedding: List[float], embeddings: List[List[float]], distance_metric="cosine"
) -> List[List]:
    """Return the distances between a query embedding and a list of embeddings."""

    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }
    distances = [distance_metrics[distance_metric](query_embedding, embedding) for embedding in embeddings]
    return distances


def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """Return a list of indices of nearest neighbors from a list of distances."""

    return np.argsort(distances)


def get_string_recommendations(strings: list[str], query_string: int) -> list[int]:
    """Return a list of sorted nearest neighbors for a given string."""

    # get the embedding of the search string
    query_embedding = get_embedding(query_string)
    # get embeddings for all strings
    embeddings = [embedding_from_string(string) for string in strings]
    # get distances between the source embedding and other embeddings (function from utils.embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from utils.utils.embeddings_utils.py)
    return indices_of_nearest_neighbors_from_distances(distances)


names = df["name_ar"].tolist() + df["name_en"].tolist()


def print_recommendations_from_strings(
    strings: list[str], index_of_source_string: int, k_nearest_neighbors: int = 1, model=EMBEDDING_MODEL
) -> list[int]:
    """Print out the k nearest neighbors of a given string."""

    # get embeddings for all strings
    embeddings = [embedding_from_string(string, model=model) for string in strings]

    # get the embedding of the source string
    query_embedding = embeddings[index_of_source_string]

    # get distances between the source embedding and other embeddings (function from utils.embeddings_utils.py)
    distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")

    # get indices of nearest neighbors (function from utils.utils.embeddings_utils.py)
    indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)

    # print out source string
    query_string = strings[index_of_source_string]
    print(f"Source string: {query_string}")
    # print out its k nearest neighbors
    k_counter = 0
    for i in indices_of_nearest_neighbors:
        # skip any strings that are identical matches to the starting string
        if query_string == strings[i]:
            continue
        # stop after printing out k articles
        if k_counter >= k_nearest_neighbors:
            break
        k_counter += 1

        # print out the similar strings and their distances
        print(
            f"""
        --- Recommendation #{k_counter} (nearest neighbor {k_counter} of {k_nearest_neighbors}) ---
        String: {strings[i][:50]}
        Distance: {distances[i]:0.3f}"""
        )

    return indices_of_nearest_neighbors
