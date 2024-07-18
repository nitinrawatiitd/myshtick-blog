---
title: "Understanding semantic search"
date: 2023-09-23
author: Nitin
tags: [llm]
---

## What is semantic search?

Semantic search or vector search is basically searching for similar meaning words/sentences/documents to your input query words/sentences/documents, instead of just pure exact match. And it is not limited to just text. It is applicable to other modalities like images, audio and video. 

In this post we'll focus exclusively on text. But how do you embed meaning into text? Well, using embeddings, which are basically high dimensional vectors (hence the term vector in vector search)

## History of embeddings

[Note: Excerpts directly picked from https://www.pinecone.io/learn/series/nlp/dense-vector-embeddings-nlp/#Dense-vs-Sparse-Vectors]

### Dense vs Sparse Vectors
The first question we should ask is why should we represent text using vectors? The straightforward answer is that for a computer to understand human-readable text, we need to convert our text into a machine-readable format.

We also have two options for vector representation; sparse vectors or dense vectors. 
* Sparse vectors is basically one hot encoded represntation of the tokens, building blocks into which we break up the text. They are sparse because they are big vectors (as big as the vocabulary i.e, all unique tokens to represent say all english text) and the values they contain are mostly 0s. And they don't carry any semantic meaning in that representation, just the syntaxt i.e, the count of tokens that make up the text.
* Dense vectors on the hand are intended to be, as the name suggest, dense. They are compact in size, typically 768 to 4096 in size, they have numerical values throughout the vector and carry semantic meaning

![](/img/sparse_vs_dense.png "Sparse vs Dense")

So dense embeddings is what we are interested in. But how do we derive these embeddings

### Evolution of word embeddings
Although we now have superior technologies for building embeddings, no overview on dense vectors would be complete without word2vec. Although not the first, it was the first widely used dense embedding model thanks to (1) being very good, and (2) the release of the word2vec toolkit — allowing easy training or usage of pre-trained word2vec embeddings.

Given a sentence, word embeddings are created by taking a specific word (translated to a one-hot encoded vector) and mapping it to surrounding words through an encoder-decoder neural net.

### Word to sentence embeddings

Why bi-encoders and not cross-encoders?
Cross encoders are slower as each comparison requires running inference which is computationally expensive

sBERT

Contrastive loss

Training Data Format

Distance measures, dot product, cosine similarity and euclidean.

Semantic search methods. Different ANN methods, what is FAISS. What is indexing mean for these methods

FAISS (From Facebook)
Capabilities on GPU vs CPU

ANNOY (Spotify)

Reranking models
Cross encoder basically that give you better comparison between shortlisted docuemnts and the input text.

