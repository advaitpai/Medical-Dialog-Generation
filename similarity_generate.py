from sentence_transformers import SentenceTransformer,util
import json
import torch
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm

def create_embeddings(sentences,batch_size,progress=True,multi=False):
    print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    # model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device='mps')
    model = SentenceTransformer('all-MiniLM-L6-v2',device = 'mps')
    embeddings = model.encode(sentences,show_progress_bar=progress,batch_size=batch_size)
    return embeddings

if __name__ == "__main__":
    df = pd.read_pickle('embeddings.pkl')
    print(df.head(10))
    user_inp = ""
    while user_inp!='0':
        user_inp = input("Enter a sentence: ")
        embedding = create_embeddings([user_inp],batch_size=1).tolist()[0]
        cos_scores = []
        for i in tqdm.tqdm(range(len(df))):
            cos_scores.append(cosine_similarity(np.array(df['patient_embeddings'].iloc[i]).reshape(1,-1),np.array(embedding).reshape(1,-1))[0][0])
        df['cosine_scores'] = cos_scores
        # df.sort_values('cosine_scores',ascending=False).iloc[:10].to_csv('output.csv')
    