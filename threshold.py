import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import numpy as np

def find_top_k_cosine(k,query_embedding,embeddings_all):
    cos_scores = []
    for i in range(len(embeddings_all)):
        cos_scores.append(cosine_similarity(np.array(embeddings_all['patient_embeddings'].iloc[i]).reshape(1,-1),np.array(query_embedding).reshape(1,-1))[0][0])
    embeddings_all['cosine_scores'] = cos_scores
    # embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:10].to_csv('output.csv')
    top_k = embeddings_all.sort_values('cosine_scores',ascending=False).iloc[:k]
    # resps = top_k['doctor_dialog']
    # print(top_k[['doctor_dialog','cosine_scores']])
    # return resps
    return top_k['cosine_scores']

def find_threshold(df_all,df_samples):
    max_cos_scores = []
    median_cos_scores = []
    min_cos_scores = []
    for i in tqdm.tqdm(range(len(df_samples))):
            scores = find_top_k_cosine(k=10,query_embedding=df_samples['patient_embeddings'].iloc[i],embeddings_all=df_all)
            max_cos_scores.append(np.max(scores))
            median_cos_scores.append(np.median(scores))
            min_cos_scores.append(np.min(scores))

    df_samples['max_cos_scores'] = max_cos_scores
    df_samples['median_cos_scores'] = median_cos_scores
    df_samples['min_cos_scores'] = min_cos_scores

    return df_samples


df_all = pd.read_pickle('datasets/embeddings/embeddings.pkl')
print(df_all)
## Comment the below two line if you want to run the code on the entire dataset
# df_samples = df_all.sample(22000,random_state=2702)# Comment this line if you want to run the code on the entire dataset
# df_all = df_samples # Comment this line if you want to run the code on the entire dataset
df_samples = df_all.sample(500,random_state=2702)
df_all = df_all.drop(df_samples.index)
df_samples = find_threshold(df_all,df_samples)
df_samples = df_samples[['max_cos_scores','median_cos_scores','min_cos_scores']]
df_samples.to_csv('datasets/threshold_large.csv')
