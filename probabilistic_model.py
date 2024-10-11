import os
import re
from collections import defaultdict
import math

# Preprocessing function
def preprocess(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load documents
def load_documents(folder_path):
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                docs[filename] = preprocess(file.read())
    return docs

# Compute term frequencies and document frequencies
def compute_statistics(docs):
    doc_freqs = defaultdict(lambda: defaultdict(int))
    doc_lengths = {}
    N = len(docs)
    avg_doc_length = 0

    for doc_id, words in docs.items():
        doc_lengths[doc_id] = len(words)
        avg_doc_length += len(words)
        for word in words:
            doc_freqs[doc_id][word] += 1
    
    avg_doc_length /= N
    return doc_freqs, doc_lengths, avg_doc_length, N

# BM25 scoring function
def bm25_score(query, doc_freqs, doc_lengths, avg_doc_length, N, k1=1.5, b=0.75):
    scores = {}
    for doc_id, doc in doc_lengths.items():
        score = 0
        for term in query:
            term_freq = doc_freqs[doc_id].get(term, 0)
            if term_freq == 0:
                continue
            doc_freq = sum(1 for doc in doc_freqs if term in doc_freqs[doc])
            idf = math.log((N - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            score += idf * ((term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * (doc_lengths[doc_id] / avg_doc_length))))
        scores[doc_id] = score
    return scores

# BM25 search function
def search_bm25(query, documents, doc_freqs, doc_lengths, avg_doc_length, N):
    preprocessed_query = preprocess(query)
    scores = bm25_score(preprocessed_query, doc_freqs, doc_lengths, avg_doc_length, N)
    ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked_docs

