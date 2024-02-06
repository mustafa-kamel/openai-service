from .utils import get_string_recommendations, names

# Using command line arguments
# import sys
# nearest_neighbors = get_string_recommendations(names[:3], sys.argv[2])[: int(sys.argv[1])]


def get_recommendations(query_string: str, recommendations_number: int = 3) -> dict:
    nearest_neighbors = get_string_recommendations(names, query_string)
    recommendations = {"query_string": query_string, "recommendations": []}
    for i in nearest_neighbors[:recommendations_number]:
        recommendations["recommendations"] += [names[i]]
    return recommendations
