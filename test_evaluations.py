# import evaluate
# rouge = evaluate.load('rouge')
# predictions = ["Transformers Transformers are fast plus efficient", 
#                "Good Morning", "I am waiting for new Transformers"]
# references = [
#               ["HuggingFace Transformers are fast efficient plus awesome", 
#                "Transformers are awesome because they are fast to execute"], 
#               ["Good Morning Transformers", "Morning Transformers"], 
#               ["People are eagerly waiting for new Transformer models", 
#                "People are very excited about new Transformers"]

# ]
# results = rouge.compute(predictions=predictions, references=references)
# print(results)

import pandas as pd
import numpy as np
import pickle

test_samples = pd.read_pickle("datasets/test_samples.pkl").reset_index(drop=True)
generated_responses = pd.read_csv("datasets/responses_all.csv")

patient_dialogs = test_samples['patient_dialog']
doctor_dialogs = test_samples['doctor_dialog']
responses = generated_responses['response']
context = generated_responses['context']
avg_cosine_scores = generated_responses['avg_cosine_scores']

df = pd.DataFrame({'patient_dialog': patient_dialogs, 'doctor_dialog': doctor_dialogs, 'response': responses, 'context': context, 'avg_cosine_scores': avg_cosine_scores})

df.to_csv("datasets/df.csv", index=False)
print("File saved!")