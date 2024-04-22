#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 08:27:57 PM EDT 2024
author: Ryan Hildebrandt, github.com/ryancahildebrandt
"""
# imports
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
from tensorflow.saved_model import save
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v1")
model.save("./minilm_model/")

use = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
save(use, "./use_model/")
