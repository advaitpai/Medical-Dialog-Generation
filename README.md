# Medical-Dialog-Generation

This GitHub repository contains the code for the project undertaken by my teammates and me as part of our CS 532 project deliverable. The report in this repository explains our motivation, methodology and an overview of our findings throughout our project. 

The datasets are available at - https://github.com/UCSD-AI4H/Medical-Dialogue-System . We use the MedDialog English dataset to create the vector embeddings. We do not publish this dataset or the created vector embeddings.

1. embeddings.py is used to create the vector embeddings.
2. threshold.py and threshold_plot.py are used to determine the threshold for cosine similarity scores for a context match.
3. similarity_generate.py is used to generate the repsonses from ChatGPT 3.5 Turbo using the augmented contexts.
4. evaluations.py is used to perform evaluation on the generated responses.


