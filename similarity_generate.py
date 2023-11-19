from sentence_transformers import SentenceTransformer,util
import json
import torch
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm
from transformers import pipeline, set_seed

global torch_device, keys, pipe
torch_device = 'cuda' # should be 'cuda' for vm or 'mps' for macbook with MX chips
with open('datasets/keys.json') as f:
    keys = json.load(f)

def create_embeddings(sentences,batch_size,progress=True,multi=False):
    # print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    # model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device=torch_device) # Use if using embeddings_large
    model = SentenceTransformer('all-MiniLM-L6-v2',device = torch_device) # Use if using embeddings
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
    # pipe = pipeline("question-answering", model="t5-large", max_new_tokens=2000, device=torch_device, token=keys["hugging_face_api_token"])
    # pipe = pipeline("question-answering", model="meta-llama/Llama-2-7b-chat-hf",device = torch_device,token=keys["hugging_face_api_token"])
    set_seed(1711)
    context_str = ""
    for i in context:
        context_str = " "+i
    # query_str = "Query - " + query + "\nContext -" + context
    # generated_text = generator("query-")
    generated_text = pipe(question = query, context = context_str)
    # generated_text = context_str
    return generated_text
    # pass


if __name__ == "__main__":
    global embeddings_all
    pipe = pipeline("question-answering", model="cxllin/Llama2-7b-med-v1", max_new_tokens=512, device=torch_device, token=keys["hugging_face_api_token"])
    print("Model succesfully loaded, loading embeddings..")
    embeddings_base_path = "datasets/embeddings/" # Should point to the relative folder containing the embeddings
    embeddings_all = pd.read_pickle(embeddings_base_path+'embeddings.pkl') # Either use embeddings.pkl or embeddings_large.pkl
    print(embeddings_all.head(10))
    user_inp = ""
    while user_inp!='0':
        user_inp = input("Enter a sentence: ")
        embedding = create_embeddings([user_inp],batch_size=1).tolist()[0]
        responses = find_top_k_responses(k=10,query_embedding=embedding)
        llm_response = fetch_llm_response(user_inp,responses)
        print(llm_response)

       