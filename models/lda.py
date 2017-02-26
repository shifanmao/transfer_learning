from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy as np
# Reference: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

def gensim_lda(docs):
    np.random.seed(seed = 12)
    
    # tokenize text
    texts = []
    for doc in docs:
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(doc)
        texts.append(tokens)

    # convert into bag-of-words
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # apply lda model
#    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=50, id2word = dictionary, passes=20)
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=50, id2word = dictionary, update_every=1, chunksize=10000, passes=1)

    return dictionary, corpus, ldamodel

def get_doc_topics(lda, bow):
    gamma, _ = lda.inference([bow])
    topic_dist = gamma[0] / sum(gamma[0])  # normalize distribution
    return [(topicid, topicvalue) for topicid, topicvalue in enumerate(topic_dist)]

def get_doc_topic(lda, bow):
    topic_inds = [topic for topic, _ in lda[bow]]
    tags = []
    for topic_ind in topic_inds:
        word_ind, _ = lda.get_topic_terms(topic_ind, topn=1)[0]
        tag = dict[word_ind]
        tags.append(tag)
    return tags, topic_inds

# collect texts
df = pd.read_csv('../output/biology_light.csv')
df["tags"] = df["tags"].map(lambda x: x.split())
num_posts = df["content"].size
posts = []
true_tags = []
for i in range(num_posts):
    posts.append(df["content"][i])
    true_tags.append(df["tags"][i])

# build lda model
dict, corpus, lda = gensim_lda(posts[:100])

# predict topics
#for i in range(len(corpus)):
for i in range(10):
    print(i, get_doc_topic(lda, corpus[i]), true_tags[i])
          
#doc1 = 'I am telling you that you are awesome,'
#doc2 = 'and this is just a test. I mean, seriously,'
#doc3 = 'I am seriously happy. And you are a little bit strange.'
