from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import wordnet as wnet
from nltk.corpus import sentiwordnet as swnet
from nltk.corpus import wordnet



class SentimentAnalysis:
    data = ""
    X = []
    Y = []
    trainX = []
    trainY = []
    testX = []
    testY = []


    def __init__(self, file="Movies_TV.txt"):
        corpus = open(file).read()
        self.data = corpus.split('\n')
        self.data.remove(self.data[0])
        self.data.remove(self.data[-1])
        for row in self.data:
            label = row[10:13]
            name = row[16:]
            self.X.append(name)
            self.Y.append(label)

    def countVectorizerTrainer(self):
        vec = CountVectorizer()
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.7, test_size = 0.3)


    def tfidfVectorizerTrainer(self):
        vec = TfidfVectorizer()
        matrix_X = vec.fit_transform(self.X)
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(matrix_X, self.Y, shuffle = True, train_size = 0.7, test_size = 0.3)


    def treeClassifier(self):
        dtc = DecisionTreeClassifier()
        dtc.fit(self.trainX, self.trainY)
        labels = dtc.predict(self.testX)
        print('Accuracy Tree classifier: ', accuracy_score(self.testY, labels))


    def linearClassifier(self):
        lc = SGDClassifier()
        lc.fit(self.trainX, self.trainY)
        labels = lc.predict(self.testX)
        print('Accuracy linear: ', accuracy_score(self.testY, labels))

    

    def KNClassifier(self):
        knn = KNeighborsClassifier()
        knn.fit(self.trainX, self.trainY)
        labels = knn.predict(self.testX)
        print('Accuracy KNeighbours: ', accuracy_score(self.testY, labels))

    def NBClassifier(self):
        nbc = MultinomialNB()
        nbc.fit(self.trainX, self.trainY)
        labels = nbc.predict(self.testX)
        print('Accuracy MultinomialNB: ', accuracy_score(self.testY, labels))





    def sentiWordNetAnalysis(self):
        final_labels = []
        for i in self.X:
            tokens = i.split(' ')
            pos_total = 0
            neg_total = 0
            for t in tokens:
                syn_t = wnet.synsets(t)
                if len(syn_t) > 0:
                    syn_t = syn_t[0]
                    senti_syn_t = swnet.senti_synset(syn_t.name())
                    if senti_syn_t.pos_score() > senti_syn_t.neg_score():
                        pos_total += senti_syn_t.pos_score()
                    else:
                        neg_total += senti_syn_t.neg_score()
            if pos_total > neg_total:
                final_labels.append("POS")
            elif pos_total < neg_total:
                final_labels.append("NEG")
            else:
                final_labels.append('NEU')
        print('Accuracy Senti_word_net_Analysis : ', accuracy_score(self.Y, final_labels))

    
    def wordNetAnalysis(self):
        good = wordnet.synsets('good')[0].name()
        good = wordnet.synsets(good)
        bad = wordnet.synsets('bad')[0].name()
        bad = wordnet.synsets(bad)
        final_labels = []
        for i in self.X:
            tokens = i.split(' ')
            pos_total = 0
            neg_total = 0
            for t in tokens:
                syn_t = wordnet.synsets(t+'.n.01')
                if len(syn_t) > 0:
                    # syn_t = syn_t[0].name()
                    # syn_t = wordnet.synsets(syn_t)
                    score = (good.wup_similarity(syn_t)) - (bad.wup_similarity(syn_t))
                    if score > 0:
                        pos_total += score
                    else:
                        neg_total += score
            if pos_total > neg_total:
                final_labels.append("POS")
            elif pos_total < neg_total:
                final_labels.append("NEG")
            else:
                final_labels.append('NEU')
        print('Accuracy Word_net_Analysis : ', accuracy_score(self.Y, final_labels))


if __name__ == "__main__":




    print("-------------")
    l1 = SentimentAnalysis()
    l1.tfidfVectorizerTrainer()
    l1.treeClassifier()
    l1.NBClassifier()
    l1.KNClassifier()
    l1.sentiWordNetAnalysis()
    l1.wordNetAnalysis()

