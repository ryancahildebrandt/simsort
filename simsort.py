#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:58:03 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
import argparse
import json

from utils import single_simsort

parser = argparse.ArgumentParser(
	prog = "SimSort",
	description = "Sort a list of texts such that semantically similar texts are closer together. Reads texts from a json file texts.json {'texts' : ['text1', 'text2', ...]} and outputs to outputs.json with structure {'texts': ['text1', 'text2', ...], 'index': [index1, index2, ...], 'sorted': ['sorted_text1', 'sorted_text2', ...]}"
	)
parser.add_argument("-f", "--filename", default = "texts.json", required = False, help = "json file to read texts from. Assumes texts are stored in json array with field name 'texts', i.e.: {'texts' : ['text1', 'text2', ...]}")
parser.add_argument("-o", "--outfile", default = "outputs.json", required = False, help = "json file to write sorted texts and indices to. Output has structure {'texts': ['text1', 'text2', ...], 'index': [index1, index2, ...], 'sorted': ['sorted_text1', 'sorted_text2', ...]}")
parser.add_argument("-e", "--embedder", default = "use", choices = ["tfidf", "minilm", "use"], required = False, help = "Embedding model to apply to texts. One of use (Universal Sentence Encoder), tfidf (Term Frequency-Inverse Document Frequency), minilm (Sentence Transformers all-MiniLM-L6-v1)")
parser.add_argument("-s", "--solver", default = "tsp_c", choices = ["lle", "mds", "pca", "isomap", "tsp_c", "tsp_g"], required = False, help = "Solver (dimension reduction or shortest path algorithm) to use when calculating sort order. One of lle (Locally Linear Embedding), mds (MultiDimensional Scaling), pca (Principal Component Analysis), isomap (Isomap), tsp_c (Traveling Salesman Shortest Path, Christofides Approximation), tsp_g (Traveling Salesman Shortest Path, Greedy Approximation)")
parser.add_argument("-t", "--test", action = "store_true", required = False, help = "Run in test mode to evaluate sort options and quality. Passes provided texts to each combination of available embedders and solvers and writes results to test_results.json")

args = parser.parse_args()

if args.test:
	test = {}

	for embedder in ["use", "tfidf", "minilm"]:
		for solver in ["lle", "isomap", "mds", "pca", "tsp_c", "tsp_g"]:
			name = f"{solver}.{embedder}"
			args.solver = solver
			args.embedder = embedder
			res = single_simsort(args)
			test[name] = res

	with open("test_results.json", "w") as f:
		f.write(json.dumps(test))
else:
	single_simsort(args)

