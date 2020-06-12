from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import string
from nltk import ngrams
import pprint
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

data = open('Movies_TV.txt').read()


data = re.sub(r'Domain.*\n', '', data)
rows = data.split('\n')
rows.remove(rows[-1])
stop_words = set(stopwords.words('english'))

inputData, total_tokens = [], []
for row in rows:
    domain, label, rating, review = row.split('\t')
    # converted into lowercase
    review = review.lower()
    
    # remove numbers from string
    review = re.sub(r'\d+', '', review)
    
    
    #removing punctuation
    review = review.translate(str.maketrans("","", string.punctuation)) 
    
    #removing spaces
    review = review.strip()  
    
    
    # tokenization and removing stopwords 
    tokens = word_tokenize(review)
    total_tokens.append([i for i in tokens if not i in stop_words])
    inputData.append(review)





#stemming
stemmer= PorterStemmer()
stemmed_tokens = []
for i in total_tokens:
    for j in i:
        temp = stemmer.stem(j)
        stemmed_tokens.append(temp)
print("\tstemmed Tokens: "stemmed_tokens)


# lemitizing
lemmatizer=WordNetLemmatizer()
lemmitized_tokens = []
for i in total_tokens:
    for j in i:
        temp = lemmatizer.lemmatize(j)
        lemmitized_tokens.append(temp)
print("\tLemitize Tokens: "lemmitized_tokens)


unique_tokens = []
unique_count = 0
count_tokens = 0
for i in total_tokens:
    count_tokens += len(i)
    unique_tokens = set(i)
    unique_count += len(unique_tokens)
print("Unique Word: "unique_count)
print("Total words: "count_tokens)



#Average length of the review
average_length_review = count_tokens/len(total_tokens)
print("Average length of the reveiws: "average_length_review) 