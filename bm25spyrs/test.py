from bm25spyrs import Retriever

documents = [
    "sustainable energy development in modern cities",
    "renewable energy systems transform cities today",
    "sustainable urban development transforms modern infrastructure",
    "future cities require sustainable planning approach",
    "energy consumption patterns in urban areas"
]

bm25 = Retriever(1.5, 0.75)
bm25.index(documents)

results = bm25.top_n("modern cities", 3)
print(results)
