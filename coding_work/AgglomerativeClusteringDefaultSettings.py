from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering

corpus = open('dataset.txt').read()



docs = corpus.split('\n')
X = []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())

vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

aggClus = AgglomerativeClustering()
aggClus.fit(matrix_X.toarray())

print(aggClus.labels_)
