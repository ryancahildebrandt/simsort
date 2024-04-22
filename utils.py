#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:57:58 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import numpy as np
import networkx as nx
import argparse
from sklearn.manifold import LocallyLinearEmbedding, Isomap, MDS
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow_hub as hub

algo_key = {
	"lle" : {"solver" : LocallyLinearEmbedding, "type" : "sklearn"},
	"isomap" : {"solver" : Isomap, "type" : "sklearn"},
	"mds" : {"solver" : MDS, "type" : "sklearn"},
	"pca" : {"solver" : PCA, "type" : "sklearn"},
	"tsp_c" : {"solver" : nx.approximation.christofides, "type" : "networkx"},
	"tsp_g" : {"solver" : nx.approximation.greedy_tsp, "type" : "networkx"},
	}

embedder_key = {
    "use" : hub.load("./use_model"),
    "tfidf" : TfidfVectorizer(),
    "minilm" : SentenceTransformer("./minilm_model"),
}

def embed_texts(texts: list, embedder: str) -> np.ndarray:
  """
	Embed texts with specified embedding model

	Args:
		texts (list): Texts to embed
		embedder (str): Embedding model

	Returns:
		np.ndarray: Embedding vectors for texts
	"""
  model = embedder_key[embedder]

  if embedder == "use":
    out = model(texts)

  if embedder == "tfidf":
    out = model.fit_transform(texts).toarray()

  if embedder == "minilm":
    out = model.encode(texts)

  return out

def create_graph_from_embeddings(embeddings: np.ndarray) -> nx.Graph:
	"""
	Create weighted graph of distances between embedded texts, where texts correspond to nodes and pairwise distances are used as edge weights

	Args:
		embeddings (np.ndarray): Embedding vectors for texts

	Returns:
		nx.Graph: Weighted pairwise distance graph
	"""
	distances = pairwise_distances(embeddings)
	pair_indices = np.ndindex(distances.shape)
	distance_values = distances.flatten()
	edges = [(inds[0], inds[1], dist) for inds, dist in zip(pair_indices, distance_values)]

	out = nx.Graph()
	out.add_weighted_edges_from(edges)

	return out

def calculate_sklearn_indices(embeddings: np.ndarray, dimension_reducer) -> list:
	"""
	Get sort indices using sklearn dimension reducer

	Args:
		embeddings (np.ndarray): Embedding vectors for texts
		dimension_reducer: sklearn.manifold embedder, used to reduce the dimensionality of provided embedding vectors to 1

	Returns:
		list: Sort indices for embedding array
	"""
	reduced_embeddings = dimension_reducer(n_components = 1).fit_transform(embeddings).flatten()
	out = np.argsort(reduced_embeddings).tolist()
	return out

def calculate_networkx_indices(graph: nx.Graph, path_approximator) -> list:
	"""
	Get sort indices using networkx shortest path approximator

	Args:
		graph (nx.Graph): Graph representation of pairwise distances between embedded texts
		path_approximator: networkx.approximation algorithm, used to calculate the shortest path between all nodes of the graph (see: traveling salesman problem)

	Returns:
		list: Sort indices for distance graph
	"""
	return nx.approximation.traveling_salesman_problem(G = graph, cycle = False, method = path_approximator)

def single_simsort(args: argparse.Namespace) -> dict:
	"""
	Run a single iteration of the simsort algorithm in simsort.py, using provided arguments (parsed from command line flags)

	Args:
		args (argparse.Namespace): User inputs

	Returns:
		dict: Results of simsort algorithm, including original texts, sort indices, and sorted texts
	"""
	algo = algo_key[args.solver]

	with open(args.filename, "r") as f:
		texts = json.load(f)["texts"]
		out = {"texts" : texts}
		print(f"Imported texts from {args.filename}")

	print(f"Sorting texts with embedder '{args.embedder}' and solver '{args.solver}'")

	embs = embed_texts(texts, args.embedder)

	if algo["type"] == "sklearn":
		index = calculate_sklearn_indices(embs, algo["solver"])
		out["index"] = index
		out["sorted"] = [texts[i] for i in index]

	if algo["type"] == "networkx":
		graph = create_graph_from_embeddings(embs)
		index = calculate_networkx_indices(graph, algo["solver"])
		out["index"] = index
		out["sorted"] = [texts[i] for i in index]

	if args.outfile:
		with open(args.outfile, "w") as f:
			f.write(json.dumps(out))

	print(f"Saving results to {args.outfile}")
	print(out)
	return out
