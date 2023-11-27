import pickle
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def tokenize(text):
    return text.split()

def bleu_score(context_tokens, response_tokens):
    scores = []
    smoothing_function = SmoothingFunction().method1
    
    for i in range(len(context_tokens)):
        try:
            # Calculate BLEU score
            score = corpus_bleu([context_tokens[i]], [response_tokens[i]], smoothing_function=smoothing_function)
            scores.append(score)
        except KeyError as e:
            # print(f"Error calculating BLEU score for sample {i}: {e}")
            continue # Skip this sample

        avg_bleu_score = sum(scores)/len(scores)
    
    return avg_bleu_score

if __name__ == "__main__":

    test_samples = pd.read_pickle("datasets/test_samples.pkl").reset_index(drop=True)
    test_samples = test_samples.iloc[0:100]
    responses = pd.read_csv("datasets/responses_advait.csv")

    # print(test_samples['doctor_dialog'].head(10))
    # print('---------------------')
    # print(responses['response'].head(10))

    # # Testing BLEU function
    # test_context = (["The cat is on the mat.", "I love coding in Python.","The weather is sunny today."])
    # test_response = (["The cat is sitting on the rug.", "I enjoy programming with Python.","Today, the weather is nice and sunny."])
    # tokenized_context = [tokenize(c) for c in test_context]
    # tokenized_response = [tokenize(r) for r in test_response]
    # print(tokenized_context)
    # print(tokenized_response)
    # b_score = bleu_score(tokenized_context, tokenized_response)
    # print(f"BLEU Score: {b_score}")

    # Calculate BLEU score for test set
    tokenized_context = [tokenize(c) for c in test_samples['doctor_dialog']]
    tokenized_response = [tokenize(r) for r in responses['response']]
    b_score = bleu_score(tokenized_context, tokenized_response)
    print(f"BLEU Score: {b_score}")

    # # Calculate BERT score
    # P, R, F1 = bert_score(context, response)
    # print(f"BERT Score: {F1}")
