# SimSort
## *Using dimensionality reduction and graphs to group similar texts*

---

## Purpose

This project provides several implementations (command line accessible python script and configurable notebook) of a simple sort function that groups semantically similar texts together. This algorithm uses some common text embedding and clustering techniques to arrange texts by similarity on a single dimension, resulting in a sorted list of texts much like sorting alphabetically or by text length. Simsort specifically targets instances where it is useful to group similar texts together (as in traditional higher dimensional clustering), but the groups should also be somehow arranged in relation to one another (i.e. sorted in some interpretable way).

---

## Approach
The simsort algorithm takes a series of unordered texts and groups them such that similar texts are together in the resulting output. Traditional sort methods are based on some absolute metric (alphabetical order, number of tokens/characters, presence of given token/character), and have predictable endpoints based on the sort criterion (a -> z, low numbers -> high numbers, short texts -> long texts). In this way, we can think of sorting techniques as placing all data observations on a 1-dimensional line, with the position of an observation corresponding to its relative similarity to one of the two endpoints of the line.

Using this framework, simsort defines the endpoints of the sorting axis as the two most dissimlar observations, with every other observation existing somewhere in between the two endpoints. Similarity is defined using the distance between the embedding vectors for each text, and these embedding vectors are projected onto a 1-dimensional space via 1) dimensionality reduction or 2) solving for the shortest path visiting all nodes of a graph based on distances between embeddings.

Simsort seeks to provide a more powerful and hopefully useful sort utility for working with text data, especially in graphical tools like spreadsheets for manual data analysis.

### Algorithm
Steps for the simsort algorithm are as follows:
- Embed unsorted texts using model of choice
- Pass embedding array into either:
  - A dimensionality reduction algorithm, bringing the dimensionality of the embeddings to n = 1 and
  - A weighted graph of pairwise distances between embedding vectors, using a shortest path algorithm to find the shortest path that visits all nodes in the graph
- Sort the original texts by either the sorted embedding values or shortest path indices

### Embedding Models
Most embedding approaches will work in the simsort algorithm, but the present implementation uses the following models:
- [Universal Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder)
- [Term Frequency-Inverse Document Frequency](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Sentence Transformers all-MiniLM-L6-v1](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v1)

### Order Solvers
The field of applicable dimensionality reduction techniques/shortest path algorithms is more limited, and most are used in the present project:
- [Locally Linear Embedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html)
- [MultiDimensional Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
- [Principal Component Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html)
- [Traveling Salesman Shortest Path, Christofides Approximation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.christofides.html)
- [Traveling Salesman Shortest Path, Greedy Approximation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.greedy_tsp.html)

---

## Outputs
- [Simsort script](./simsort.py) *Note:* If you're running this on your local machine, you may need to run download_models.py to populate embedding model files
- [Simsort notebook](./simsort.ipynb)
