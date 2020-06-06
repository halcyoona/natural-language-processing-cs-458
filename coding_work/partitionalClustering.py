from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

corpus = open('E:\\dataset.txt').read()
docs = corpus.split('\n')
X = []
for doc in docs:
    i, l = doc.split(':')
    X.append(i.strip())

vec = CountVectorizer()
matrix_X = vec.fit_transform(X)

kmeans = KMeans(n_clusters = 2, max_iter = 300, tol = 1e-4)
kmeans.fit(matrix_X[:5])

print(kmeans.labels_)
print(kmeans.predict(matrix_X[5]))


kn = NearestNeighbors()
kn.fit(matrix_X)

print(kn.kneighbors(matrix_X[3], 2))
kn.radius_neighbors(matrix_X[3], radius = 1.7)

