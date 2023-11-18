from sentence_transformers import SentenceTransformer,util
import json
import torch
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm
from transformers import pipeline, set_seed

def create_embeddings(sentences,batch_size,progress=True,multi=False):
    print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device='mps') # Embedding Large
    # model = SentenceTransformer('all-MiniLM-L6-v2',device = 'mps') # Embedding small
    embeddings = model.encode(sentences,show_progress_bar=progress,batch_size=batch_size)
    return embeddings

def find_top_k_responses(k,query_embedding):
    cos_scores = []
    for i in tqdm.tqdm(range(len(embeddings_all))):
        cos_scores.append(cosine_similarity(np.array(embeddings_all['patient_embeddings'].iloc[i]).reshape(1,-1),np.array(query_embedding).reshape(1,-1))[0][0])
    embeddings_all['cosine_scores'] = cos_scores
    # embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:10].to_csv('output.csv')
    top_k = embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:k]
    resps = top_k['doctor_dialog']
    return resps

def fetch_llm_response(query,context):
    pass


if __name__ == "__main__":
    global embeddings_all
    embeddings_all = pd.read_pickle('embeddings_large.pkl')
    print(embeddings_all.head(10))
    user_inp = ""
    while user_inp!='0':
        user_inp = input("Enter a sentence: ")
        embedding = create_embeddings([user_inp],batch_size=1).tolist()[0]
        responses = find_top_k_responses(k=10,query_embedding=embedding)
        llm_response = fetch_llm_response(user_inp,responses)
        print(llm_response)

       