import os
import pandas as pd
import numpy as np
# import json
# import tiktoken
import pickle as pkl
import psycopg2
# import ast
import pgvector
import torch
from sentence_transformers import SentenceTransformer
import math
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

# df = pd.read_csv('blog_data_and_embeddings.csv')
# df['doctor_dialogue'] = df['content']
# df['doctor_embeddings'] = df['embeddings']
print(os.getcwd())
with open("/datasets/MedDialog_English/healthcaremagic_dialogue_1.json", "rb") as f:
    object = pkl.load(f)
df = pd.DataFrame(object)
print(type(df))
exit()
# print(type(df['embeddings'][0]),df.columns)
# exit()
# print(list(df['embeddings'][0]))

# exit()
# print(df.head(10))

# connection_string  = 'postgres://tsdbadmin@pyf45tatad.aklybuoun2.tsdb.cloud.timescale.com:36224/tsdb?sslmode=require'

# conn = psycopg2.connect(connection_string, )

conn = psycopg2.connect(
database="meddialog_vectordb",
user="divyasha",
password="MedicalDialogue",
host="localhost",
port= '5432'
)
cur = conn.cursor()

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Register the vector type with psycopg2
register_vector(conn)

# Create table to store embeddings and metadata
table_create_command = """
CREATE TABLE embeddings (
            id bigserial primary key, 
            link text,
            description text,
            patient text,
            pateint_embedding vector(1536),
            doctor text,
            doctor_embedding vector(1536)
            );
            """

# cur.execute(table_create_command)
# cur.close()
# conn.commit()

#Batch insert embeddings and metadata from dataframe into PostgreSQL database
register_vector(conn)
cur = conn.cursor()
# Prepare the list of tuples to insert
# data_list = [(row['id'],row['description'], row['link'], row['patient'], np.array(row['patient_embeddings']),row['doctor'], np.array(row['doctor_embeddings'])) for index, row in df.iterrows()]
data_list = [(row['title'], row['url'], row['content'],row['embeddings'],row['doctor_dialogue'],row['doctor_embeddings']) for index, row in df.iterrows()]
# print(type(data_list[0][3]))
# exit()
# insert_query = "INSERT INTO embeddings (description, link, patient, pateint_embedding, doctor, doctor_embedding) VALUES %s"

# Use execute_values to perform batch insertioncur.execute("INSERT INTO embeddings (title, url, content, embeddings,content,embedding) VALUES {}", .format(data_list[0]))
# execute_values(cur,"INSERT INTO embeddings (description, link, patient, pateint_embedding, doctor,doctor_embedding) VALUES %s",data_list)
# # cur.execute(insert_query, data_list[0])
# # Commit after we insert all embeddings
# conn.commit()

cur.execute("SELECT COUNT(*) as cnt FROM embeddings;")
num_records = cur.fetchone()[0]
print("Number of vector records in table: ", num_records,"\n")

# # print the first record in the table, for sanity-checking
# cur.execute("SELECT * FROM embeddings LIMIT 1;")
# records = cur.fetchall()
# print("First record in table: ", records)

# Create an index on the data for faster retrieval

#calculate the index parameters according to best practices
num_lists = num_records / 1000
if num_lists < 10:
   num_lists = 10
if num_records > 1000000:
   num_lists = math.sqrt(num_records)

#use the cosine distance measure, which is what we'll later use for querying
cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (pateint_embedding  vector_cosine_ops) WITH (lists = {num_lists});')
# cur.execute(f'CREATE INDEX ON embeddings USING ivfflat (doctor_embedding  vector_cosine_ops) WITH (lists = {num_lists});')
conn.commit()

#Query the database for similar vectors
input = "How is Timescale used in IoT?"

# Helper function: Get top K most similar documents from the database
def get_topK_similar_docs(query_embedding, conn):
    embedding_array = np.array(query_embedding)
    # Register pgvector extension
    register_vector(conn)
    cur = conn.cursor()
    # Get the top K most similar documents using the KNN <=> operator
    cur.execute("SELECT content FROM embeddings ORDER BY embedding <=> %s LIMIT 3", (embedding_array,))
    topK_docs = cur.fetchall()
    return topK_docs

def get_embeddings(text,progress=True,multi=False):
    print("Checking if MPS backend for Torch is available:",torch.backends.mps.is_available())
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1',device='mps')
    # model = SentenceTransformer('all-MiniLM-L6-v2',device = 'mps')
    embedding = model.encode(text,show_progress_bar=progress)
    return embedding











