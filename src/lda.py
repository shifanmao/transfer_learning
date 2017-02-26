from gensim import corpora, models
# Reference: https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

def lda(texts):
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    print corpus

test_text = ['I am telling you that you are awesome,',
             'and this is just a test. I mean, seriously,',
             'I am seriously happy. And you are a little bit strange.']

lda(test_text)
