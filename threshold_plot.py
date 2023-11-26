import pandas as pd
import numpy as np

df = pd.read_csv('datasets/threshold_large_500.csv')
print(df.head(10))

print("Mean of median cosine scores: ",np.mean(df['median_cos_scores']))
print("Median of median cosine scores: ",np.median(df['median_cos_scores']))
print("Mean of min cosine scores: ",np.mean(df['min_cos_scores']))
print("Median of min cosine scores: ",np.median(df['min_cos_scores']))
print("Min of median cosine scores: ",np.min(df['median_cos_scores']))
print("Min of min cosine scores: ",np.min(df['min_cos_scores']))

# Plotting a graph for max, median and min cosine scores
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
sns.set_context('paper')
sns.set_palette('Set2')
# plt.figure(figsize=(10,6))
# plt.plot(df['max_cos_scores'],label='Max Cosine Scores')
# plt.plot(df['median_cos_scores'],label='Median Cosine Scores')
# plt.plot(df['min_cos_scores'],label='Min Cosine Scores')
# plt.xlabel('Samples')
# plt.ylabel('Cosine Scores')
# plt.legend()
# plt.show()

# Statistical analysis for max, median and min cosine scores
print(df['min_cos_scores'].describe())

# Plotting a histogram for max, median and min cosine scores using line plot
plt.figure(figsize=(10,6))
plt.hist(df['max_cos_scores'],label='Max Cosine Scores',alpha=0.5)
plt.hist(df['median_cos_scores'],label='Median Cosine Scores',alpha=0.5)
plt.hist(df['min_cos_scores'],label='Min Cosine Scores',alpha=0.5)
plt.xlabel('Cosine Scores')
plt.ylabel('Frequency')
plt.legend()
plt.show()




# Statistical analysis for max, median and min cosine scores
print(df['median_cos_scores'].describe())