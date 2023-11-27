import pickle
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def tokenize(text):
    return text.split()

def bleu_score(test_df):

    # tokenize doctor dialogs and responses
    response_tokens = [tokenize(r) for r in test_df['response']]
    docdialogs_tokens = [tokenize(d) for d in test_df['doctor_dialog']]

    scores = []
    count = 0
    smoothing_function = SmoothingFunction().method1
    
    for i in range(len(response_tokens)):
        try:
            # if test_df['context'] is not empty calculate bleu score else skip the sample
            if test_df['context'][i] != '[]':
                # Calculate BLEU score
                score = corpus_bleu([docdialogs_tokens[i]], [response_tokens[i]], smoothing_function=smoothing_function)
                scores.append(score)
                count += 1
            else:
                continue
        except KeyError as e:
            continue
            
    print(f"BLEU score calculated for {count} out of {len(response_tokens)} samples")

    return scores

# # calculate rouge score
# def rouge_score(context, reponse):

#     context = context.tolist()
#     reponse = reponse.tolist()
#     rouge = evaluate.load("rouge")
#     scores = rouge.compute(predictions=[reponse], references=[context])
#     return scores


if __name__ == "__main__":

    test_samples = pd.read_pickle("datasets/test_samples.pkl").reset_index(drop=True)
    # extract 0 to100 and and 200 to 300 samples
    test_samples_1 = test_samples.iloc[0:100]
    test_samples_2 = test_samples.iloc[200:300]
    test_samples = pd.concat([test_samples_1, test_samples_2], ignore_index=True)
    doctor_dialogs = test_samples['doctor_dialog']
    generated_responses = pd.read_csv("datasets/responses_all.csv")
    responses = generated_responses['response']
    context = generated_responses['context']

    # create dataframe having columns doctor dialogs and responses and context
    df = pd.DataFrame({'doctor_dialog': doctor_dialogs, 'response': responses, 'context': context})

    print('length of actual doctor dialogs: ', len(doctor_dialogs))
    print('length of responses: ', len(responses))
    print('\n')

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
    tokenized_context = [tokenize(c) for c in context]
    tokenized_response = [tokenize(r) for r in responses]
    b_scores = bleu_score(df)
    
    # save to text file
    with open('datasets/bleu_scores.txt', 'w') as f:
        for item in b_scores:
            f.write("%s\n" % item)

    avg_b_score = sum(b_scores)/len(b_scores)
    print(f"Average BLEU Score: {avg_b_score}")

    # # Calculate ROUGE score for test set
    # r_score = rouge_score(context, responses)
    # print(f"ROUGE Score: {r_score}")


    # # Calculate BERT score
    # P, R, F1 = bert_score(context, response)
    # print(f"BERT Score: {F1}")
