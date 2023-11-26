from sentence_transformers import SentenceTransformer,util
import json
import torch
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tqdm
from transformers import pipeline, set_seed
import openai
import time
import csv
import json


global torch_device, keys, threshold
threshold = 0.62

torch_device = 'mps' # should be 'cuda' for vm or 'mps' for macbook with MX chips
with open('datasets/keys.json') as f:
    keys = json.load(f)
    openai.api_key = keys["openai_api_token"]

def create_embeddings(sentences,batch_size,progress=True,multi=False):
    # print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device=torch_device) # Use if using embeddings_large
    # model = SentenceTransformer('all-MiniLM-L6-v2',device = torch_device) # Use if using embeddings
    embeddings = model.encode(sentences,show_progress_bar=progress,batch_size=batch_size)
    return embeddings

def find_top_k_responses(k,query_embedding):
    cos_scores = []
    for i in tqdm.tqdm(range(len(embeddings_all))):
        cos_scores.append(cosine_similarity(np.array(embeddings_all['patient_embeddings'].iloc[i]).reshape(1,-1),np.array(query_embedding).reshape(1,-1))[0][0])
    embeddings_all['cosine_scores'] = cos_scores
    # embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:10].to_csv('output.csv')
    top_k = embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:k]
    # print(top_k[['doctor_dialog','cosine_scores']])
    resps = top_k[['doctor_dialog','cosine_scores']]
    resps, cosine_scores = filter_by_threshold(resps,threshold)
    return resps, cosine_scores

def fetch_llm_response(query,context):
    set_seed(1711)
    context_str = ""
    for i in context:
        context_str += " "+i
    
    messages = [{"role":"system",
             "content":"You are a helpful healthcare assistant. Question: %s" %query,
             },
            ]
    prompt = """Respond like a chatbot giving an extremely engaging response based on the context given below.

    context : <<CONTEXT>>

    DETAILED SUMMARY:

    """

    prompt = prompt.replace("<<CONTEXT>>",context_str)
    print(prompt)

    messages.append({"role":"user","content":prompt})


    return retreive_summary(messages)

def retreive_summary(messages):
        max_retry = 1
        retry = 0
        code = 0
        #
        while True:
            try:
                chat = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                                    messages = messages,
                                                    temperature=0,
                                                    )
                reply = chat.choices[0].message.content, code
                return reply
            except Exception as oops:
                retry += 1
                time.sleep(25)
                if retry >= max_retry:
                    code=-1
                    return "Accessing the Completion service error: %s" % oops, code
                    
#filter by threshold
def filter_by_threshold(responses,threshold):
    res = []
    cosine_scores = []
    for i in len(responses):
        if responses.iloc[i]['cosine_scores']>=threshold:
            res.append(responses.iloc[i]['doctor_dialog'])
            cosine_scores.append(responses.iloc[i]['cosine_scores'])
    return res,cosine_scores

if __name__ == "__main__":
    global embeddings_all
    embeddings_base_path = "datasets/embeddings/" # Should point to the relative folder containing the embeddings
    embeddings_all = pd.read_pickle(embeddings_base_path+'embeddings_large.pkl') # Either use embeddings.pkl or embeddings_large.pkl
    print("Embeddings fetched:",embeddings_all.head(10))

    base_path = "datasets/results/"
    file_name = "results.pkl" # Either use embeddings.pkl or embeddings_large.pkl
    
   
    
    test_set = pd.read_pickle("datasets/test_samples.pkl")
    # advait_test_set = test_set.iloc[:100]
    divyasha_test_set = test_set.iloc[100:103]
    # zohair_test_set = test_set.iloc[200:300]
    # print("Test set loaded:", [divyasha_test_set.iloc[0]['patient_dialog']])
    # exit()

    if not os.path.exists(base_path+file_name):
        resps = pd.DataFrame(columns=['message','response','avg_cosine_scores'])
        res_cnt=0
    else:
        resps = pd.read_pickle(base_path+file_name)
        res_cnt = len(resps)
    
    
    for i in range(res_cnt,len(resps)):
        start = time.process_time()
        embedding = create_embeddings([divyasha_test_set.iloc[i]['patient_dialog']],batch_size=1).tolist()[0]
        responses = find_top_k_responses(k=10, query_embedding=embedding)
        llm_response, code = fetch_llm_response(divyasha_test_set.iloc[i]['patient_dialog'], responses)
        if code == -1:
            resps.to_pickle("datasets/responses.pkl")
            print(llm_response)
            exit()
        resps.at[i, 'message'] = divyasha_test_set.iloc[i]['message']
        resps.iloc[i]['response'] = llm_response
        resps.iloc[i]['avg_cosine_scores'] = np.mean(responses['cosine_scores'])
        resps.to_pickle("datasets/responses.pkl")
        proc_time = time.process_time() - start
        print("Time: ", proc_time)
        if proc_time < 30:
            time.sleep(20)




    
    # while res_cnt!=len(messages):
    #     print("res_cnt: ", res_cnt)
    #     start = time.process_time()
    #     embedding = create_embeddings([messages.iloc[i]['patient_dialog']],batch_size=1).tolist()[0]
    #     responses = find_top_k_responses(k=10,query_embedding=embedding)
        
    #     llm_response,code = fetch_llm_response(messages.iloc[i]['patient_dialog'],responses)
    #     if code==-1:
    #         resps.to_csv("datasets/responses.csv", index=False)
    #         print(llm_response)
    #         exit()
    #     resps.at[i, 'message'] = messages.iloc[i]['message']
    #     resps.iloc[i]['response'] = llm_response
    #     # with open('datasets/responses.csv', 'a') as f:
    #     #    f.write('\n')
    #     # resps.to_csv("datasets/responses.csv",mode='a',header=False,index=False)
    #     resps.to_csv("datasets/responses.csv", index=False)
    #     # time.sleep(15)
    #     proc_time=time.process_time() - start
    #     print("Time: ",proc_time)
    #     if proc_time<30:
    #         time.sleep(15)
    #     i+=1

    

    # user_inp = ""
    # while user_inp!='0':
    #     user_inp = input("Enter a sentence: ")
    #     embedding = create_embeddings([user_inp],batch_size=1).tolist()[0]
    #     # print(embedding)
    #     responses = find_top_k_responses(k=10,query_embedding=embedding)
    #     llm_response = fetch_llm_response(user_inp,responses)
    #     print(llm_response)
