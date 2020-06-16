from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re


"""Read data from txt file"""

corpus = open('movies_data.txt').read()      



"""Remove label from starting like Domain and lebel ..."""

corpus = re.sub(r'Domain.*\n', '', corpus)


"""Extract reviews form data (courpus)"""

rows = corpus.split('\n')
rows.remove(rows[-1])
inputData, y = [], []
for row in rows:
    _, label, _, review = row.split('\t')
    inputData.append(review)
    y.append(label)


"""binary"""

vec = CountVectorizer(ngram_range = (1, 3),max_features =1000,min_df  = 10,max_df=100, binary=True) 
array = vec.fit_transform(inputData)


"""frequency"""

vec = CountVectorizer(ngram_range = (1, 3),max_features =1000,min_df  = 10,max_df=100) 
array = vec.fit_transform(inputData)

"""Structred with tdfidf"""

vec2 = TfidfVectorizer(binary=True)
array = vec2.fit_transform(inputData)

array.toarray()
