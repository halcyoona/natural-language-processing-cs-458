
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors



class Clustering:
    data = []
    def preprocessing(self, file_path):
        corpus = open(file_path).read()
        self.data = corpus.split('\n')
        self.data.remove(self.data[0])


    def meanShiftClustering(self, rad, clustr):
        vec = CountVectorizer()
        matrix_X = vec.fit_transform(self.data)

        knn = NearestNeighbors(radius = rad)
        fit = knn.fit(matrix_X)
        con = matrix_X.toarray()
        formed_cluster = MeanShift(bandwidth=clustr).fit(con)
        print(formed_cluster.labels_)

    def KMeansClustering(self, rad, clustr):
        vec = CountVectorizer()
        matrix_X = vec.fit_transform(self.data)
        knn = NearestNeighbors(radius = rad)
        fit = knn.fit(matrix_X)
        con = matrix_X.toarray()
        formed_cluster = KMeans(n_clusters = clustr, max_iter = 100, tol = 1e-4).fit(con)
        print(formed_cluster.labels_)




if __name__ == "__main__":
    C1 = Clustering()
    C1.preprocessing("Movies_TV.txt")
    C1.meanShiftClustering(10, 2)
    C1.meanShiftClustering(10, 5)
    C1.KMeansClustering(10, 3)
    C1.KMeansClustering(10, 5)