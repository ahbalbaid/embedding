# reading cnn_samples and federal_samples

import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy import spatial
import nltk
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset="train", shuffle = True)
newsgroups_test = fetch_20newsgroups(subset="test", shuffle = True)
#nltk.download('wordnet')
#nltk.download('all')
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
stemmer = SnowballStemmer("english")
np.random.seed(400)

df1 = pd.read_csv("DataBase/cnn_samples.csv")
df2 = pd.read_csv("DataBase/federal_samples.csv")
df2.rename(columns = {"embeddings": "embedding"}, inplace = True)

# fix embedding column
embed_col = []
for i in range(len(df1["embedding"])):
    embed_col.append(df1["embedding"][i])

for i in range(len(df2["embedding"])):
    embed_col.append(df2["embedding"][i])

for i in range(len(df1["embedding"])+len(df2["embedding"])):
    embed_col[i] = ast.literal_eval(embed_col[i])

df = pd.concat([df1, df2])
df["embedding"] = embed_col

# clustering data
# inertias = []
# max_clusters = 30

# for i in range(1, max_clusters + 1):
#     kmeans = KMeans(n_clusters=i, random_state=0).fit(preprocessing.normalize(df["embedding"].tolist()))
#     inertias.append(kmeans.inertia_)

# graphing inertias

# plt.plot(range(1, max_clusters + 1),inertias, '-bo')
# plt.xlabel("k values")
# plt.ylabel("inertia")
# plt.show()

# add labels to dataframe

kmeans = KMeans(n_clusters=12, random_state=0).fit(preprocessing.normalize(df["embedding"].tolist()))

if "labels" in df.columns:
    df["labels"] = kmeans.labels_
else:
    df.insert(0, "labels", kmeans.labels_)

categories = [
    "Technology",
    "Money",
    "Law",
    "Energy / Environment",
    "crime / finance / others",
    "Health",
    "Employment",
    "World News",
    "Politics",
    "Entertainment",
    "Violence",
    "finance / business / tech"
]

# import challenge articles
A = input('path of the .csv (challenge.csv)')
dfc = pd.read_csv(A)
dfc.rename(columns = {"embeddings": "embedding"}, inplace = True)

embed_col = []
for i in range(len(dfc["embedding"])):
    embed_col.append(ast.literal_eval(dfc["embedding"][i]))

dfc["embedding"] = embed_col

print("------------Topics------------")
for i in kmeans.predict(preprocessing.normalize(dfc["embedding"].tolist())):
    print(categories[i])

print("------------MostSimilarArticles------------")
# print top 3 articles most similar to challenge articles



def cos_similarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

for i in range(len(dfc["embedding"])):
    similarities = [cos_similarity(embedding, dfc["embedding"][i]) for embedding in df["embedding"]]
    top_3 = sorted(similarities, reverse=True)[0:3]

    print(top_3)
    print(df.iloc[similarities.index(top_3[0]), 3])
    print()

print("------------MostImportantWords------------")


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


# Tokenize and lemmatize
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result

processed_docs = []
for i in kmeans.predict(preprocessing.normalize(dfc["embedding"].tolist())):
    label = df[df["labels"] == i]['text']
    for doc in label:
        processed_docs.append(preprocess(doc))
    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=10, no_above=0.25, keep_n=100000 )
    if len(dictionary) < 10:
        label = dfc[dfc["labels"] == i]['text']
        for doc in label:
            processed_docs.append(preprocess(doc))
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=5, no_above=0.25, keep_n=100000 )
    count = 0
    for k, v in dictionary.iteritems():
        print(k, v,end=" ,")
        count += 1
        if count > 10:
            break
    print('\n')
    processed_docs = []