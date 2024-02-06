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


# Fields: (Uniq Id,Crawl Timestamp,Product Url,Product Name,Description,List Price,
# Sale Price,Brand,Item Number,Gtin,Package Size,Category,Postal Code,Available)
# n_examples = 5
# df.head(n_examples)
# for idx, row in df.head(n_examples).iterrows():
#     print("")
# print(f"Title: {row['Product Name']}")
# print(f"Description: {row['Description'][:30]}")
# print(f"Label: {row['Category']}")
# PRODUCTS DETAILS
# print(f"Name AR: {row['name_ar']}")
# print(f"Name EN: {row['name_en']}")
# print(f"Price: {row['price']}")
# descriptions = df["Description"][:500].tolist()
# recommended_products = print_recommendations_from_strings(
#     strings=descriptions,  # let's base similarity off of the article description
#     index_of_source_string=0,  # articles similar to the first one about Tony Blair
#     k_nearest_neighbors=5,  # 5 most similar articles
# )
# print(recommended_products[:5])
