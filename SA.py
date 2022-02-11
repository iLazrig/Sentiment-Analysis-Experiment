

import pandas as pd
import numpy as np
import re

import os

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from nltk.classify import SklearnClassifier
from sklearn.model_selection import train_test_split 

nltk.download('stopwords')


sentmntDir = "Sentiment"

fName="All-Sentiments.xlsx"
#columns= ['Answer','FinalSentiment','Sat','Dissat']
allSentmnts = pd.read_excel(fName)# ,usecols=[1,2,3,4,5])

print("total=", len(allSentmnts), " -- +ve = ",len(allSentmnts[allSentmnts['FinalSentiment']==1])," -- -ve = ",len(allSentmnts[allSentmnts['FinalSentiment']==-1]), "-- neut=", len(allSentmnts[allSentmnts['FinalSentiment']==0]) )



allSentmntsWOneutral = allSentmnts[allSentmnts['FinalSentiment']!=0]

features = allSentmnts['Answer'].values #text column
labels = allSentmnts['FinalSentiment'].values #sentiment column


def ProcessText(inText):
    processed_inText = []

    for sentence in range(0, len(inText)):

        #Remove all special characters
        processed_t = re.sub(r'\W',' ',str(inText[sentence]))

        #Remove all single characters in the middle of the sentence, start of sentence, and end of sentence
        # \D: non-digit characters      
        processed_t = re.sub(r'(\s\D\s)',' ',processed_t)

        # Remove single characters from the start
        processed_t= re.sub(r'^(\D\s)', ' ', processed_t) 

        # Substituting multiple spaces with single space
        processed_t = re.sub(r'\s+', ' ', processed_t, flags=re.I)

        
        # to Lowercase
        processed_t = processed_t.lower()

        processed_inText.append(processed_t)
    return processed_inText


processed_features = ProcessText(features)

vectorizer = TfidfVectorizer (max_features = 2500, min_df=7, max_df=0.8, stop_words = stopwords.words('english'))
Vprocessed_features = vectorizer.fit_transform(processed_features).toarray()


vocab1=vectorizer.get_feature_names()


X_train, X_test, y_train, y_test = train_test_split(Vprocessed_features, labels, test_size = 0.2, random_state = 0)


text_classifier = RandomForestClassifier(n_estimators=400, random_state=0,criterion='entropy',bootstrap=False,min_samples_split=3, verbose=2,class_weight={1: 1, 0:10,-1: 6})
text_classifier.fit(X_train, y_train)

predictions = text_classifier.predict(X_test)


print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))

X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)



def get_words_in_wDoc(wDoc):
    allWds = []
    for (words, sentiment) in wDoc:
        allWds.extend(words)
    return allWds

def doc2Words(doc):
    mtwl=0                
    w_doc = []
    pattern = re.compile('[\W_]+')
    stopwords_set = set(stopwords.words("english"))
    
    for index, row in doc.iterrows():
        words_filtered = [pattern.sub('',e.lower() )for e in row.text.split() if len(e) >= 3]
        #[e.lower() for e in row.text.split() if len(e) >= 3]
        words_cleaned = [word for word in words_filtered
            if 'http' not in word
            and not word.startswith('@')
            and not word.startswith('#')
            
            and word != 'RT']
        
        words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
        
        #twl=len(words_wo_sw_highFreqWords)
        #mtwl= twl if twl> mtwl else mtwl
        
        
        #twl=len(words_cleaned)
        #mtwl= twl if twl> mtwl else mtwl
        w_doc.append((words_cleaned,row.sentiment))
    #print(mtwl)
    
    
    wordlist=get_words_in_wDoc(w_doc)
    
    wordFreq=nltk.FreqDist(wordlist)
    
    com10=wordFreq.most_common(10)
    highFreqWords = [w[0] for w in com10]
    nw_doc=[]
    for wds,sent in w_doc:
        words_wo_sw_highFreqWords = [word for word in wds if not word in highFreqWords] 
        nw_doc.append((words_wo_sw_highFreqWords,sent))
    
    return nw_doc





oldColNames=allSentmnts.columns
allSentmnts.columns=['text','sentiment']



nw_doc = doc2Words(allSentmnts)
wordlist=get_words_in_wDoc(nw_doc)

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    features = wordlist.keys()
    return features


w_features = get_word_features(get_words_in_wDoc(nw_doc))



def find_features(document):
    words = set(document)
    features = {}
    for w in w_features:
        features[w] = (w in words)

    return features



def extract_features(document):
    document_words = set(document)
    features = {}
    for word in w_features:
        features['containts(%s)' % word] = (word in document_words)
    return features




# Splitting the dataset into train and test set
train, test = train_test_split(nw_doc,test_size = 0.2)

# Training the Naive Bayes classifier
training_set = nltk.classify.apply_features(find_features,train)


testing_set = nltk.classify.apply_features(find_features,test)


classifier = nltk.NaiveBayesClassifier.train(training_set)


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)



for txt,sent in test:
    dist=classifier.prob_classify(find_features(txt))
    print(sent), 
    for label in dist.samples():
        print("%s: %f" % (label, dist.prob(label)))
    print("\n")



# Comparing Additional Classifiers


from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


# Using scikit-learn Classifiers With NLTK


features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
    for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
])

# Use partof the dataset for training

for name, sklearn_classifier in classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set)
    print(F"{accuracy:.2%} - {name}")


# ##### Without neutral 

allSentmntsWOneutral.columns=['text','sentiment']



nw_docWon = doc2Words(allSentmntsWOneutral)
w_features = get_word_features(get_words_in_wDoc(nw_docWon))
#wordlist=get_words_in_wDoc(nw_docWon)

trainWoN, testWoN = train_test_split(nw_docWon,test_size = 0.2)


# Training the Naive Bayes classifier
trainingWoN_set = nltk.classify.apply_features(find_features,trainWoN)


testingWoN_set = nltk.classify.apply_features(find_features,testWoN)


classifierWoN = nltk.NaiveBayesClassifier.train(trainingWoN_set)



print("Classifier accuracy percent:",(nltk.classify.accuracy(classifierWoN, testingWoN_set))*100)




for name, sklearn_classifier in classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(trainingWoN_set)
    accuracy = nltk.classify.accuracy(classifier, testingWoN_set)
    print(F"{accuracy:.2%} - {name}")



# ##### Trained with other dataset

# Reading movie review dataset

from nltk.corpus import movie_reviews


cat='neg'
movieReviewsDf=pd.DataFrame()
for fileid in movie_reviews.fileids(cat):
    f = movie_reviews.open(fileid)
    txt=f.read()
    f.close()
    new_row={'text':txt,'sentiment':-1}
    movieReviewsDf=movieReviewsDf.append(new_row, ignore_index=True)
    #print(txt)
cat='pos'
for fileid in movie_reviews.fileids(cat):
    f = movie_reviews.open(fileid)
    txt=f.read()
    f.close()
    new_row={'text':txt,'sentiment':1}
    movieReviewsDf=movieReviewsDf.append(new_row, ignore_index=True)

movieReviewsDf['sentiment']=movieReviewsDf['sentiment'].astype(int)


mvTrainData=doc2Words(movieReviewsDf)
#use the features from this dataset
w_features = get_word_features(get_words_in_wDoc(mvTrainData))
#wordlist=get_words_in_wDoc(mvTrainData)


# Training the Naive Bayes classifier
mvTrainingFeat_set = nltk.classify.apply_features(find_features,mvTrainData)


testingWoN_set = nltk.classify.apply_features(find_features,testWoN)


classifier1 = nltk.NaiveBayesClassifier.train(mvTrainingFeat_set)


# since movie data has no NEUTRAL sent, we will use our data without the NEUTRAL 


print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier1, testingWoN_set))*100)



for name, sklearn_classifier in classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(mvTrainingFeat_set)
    accuracy = nltk.classify.accuracy(classifier, testingWoN_set)
    print(F"{accuracy:.2%} - {name}")


# using all of our dataset for testing
# 

testingWoN_set2 = nltk.classify.apply_features(find_features,nw_docWon)

print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier1, testingWoN_set2))*100)



for name, sklearn_classifier in classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(mvTrainingFeat_set)
    accuracy = nltk.classify.accuracy(classifier, testingWoN_set2)
    print(F"{accuracy:.2%} - {name}")


# Using airlines tweets

data_source_url = "Tweets.csv"

airline_tweets = pd.read_csv(data_source_url)

TrData=airline_tweets.iloc[:,[10,1]]
TrData[:5]

codes = {'neutral':0, 'positive':1, 'negative':-1}
TrData['airline_sentiment'] = TrData['airline_sentiment'].map(codes)
TrData[:5]



TrData.columns=['text','sentiment']


print(sum(TrData['sentiment']==1),sum(TrData['sentiment']==-1),sum(TrData['sentiment']==0))



alTrainData=doc2Words(TrData)
#use the features from this dataset
w_features = get_word_features(get_words_in_wDoc(alTrainData))
#wordlist=get_words_in_wDoc(alTrainData)


# Training the Naive Bayes classifier
alTrainingFeat_set = nltk.classify.apply_features(find_features,alTrainData)


classifier2 = nltk.NaiveBayesClassifier.train(alTrainingFeat_set)


# using all of our data for testing including the Neu

testing_set3 = nltk.classify.apply_features(find_features,nw_doc)



print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier2, testing_set3))*100)




for name, sklearn_classifier in classifiers.items():
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(alTrainingFeat_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set3)
    print(F"{accuracy:.2%} - {name}")



# pre trained model


nltk.download('vader_lexicon')


#from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer
VaSa = SentimentIntensityAnalyzer()

traindata, testdata = train_test_split(allSentmnts,test_size = 0.99)

testdata = allSentmnts


from random import shuffle

def is_positive(rev: str) -> bool:
   
    n = VaSa.polarity_scores(rev)["neu"]
    c = VaSa.polarity_scores(rev)["compound"]
    return c>=0.05 #c > 0 

def is_negative(rev: str) -> bool:
    
    return VaSa.polarity_scores(rev)["compound"] <=-0.05 #< 0 

def is_neutral(rev: str) -> bool:
    
    c = VaSa.polarity_scores(rev)["compound"]
    n = VaSa.polarity_scores(rev)["neu"]
    return  (c>-0.05 and c<0.05) #(c >= 0 and n> c)

cp=0
cn=0
cnu=0
ncp=0
ntcp=0
pcn=0
ntcn =0
ncnt=0
pcnt=0
y_true=[]
y_pred=[]
for i,r in testdata.iterrows():
    
    if is_positive(r[0]) :
        predSen = 1
        cp = cp+1 if r[1]==1 else cp
        ncp=ncp+1 if r[1] ==-1 else ncp
        ntcp=ntcp+1 if r[1] ==0 else ntcp
    elif is_negative(r[0]) :
        predSen = -1
        cn=cn+1 if r[1] ==-1 else cn
        pcn=pcn+1 if r[1] ==1 else pcn
        ntcn = ntcn+1 if r[1] ==0 else ntcn
        #cn+=1
    elif is_neutral(r[0]):
        predSen = 0
        cnu= cnu+1 if r[1]==0 else cnu
        ncnt = ncnt+1 if r[1]==-1 else ncnt
        pcnt=pcnt+1 if r[1] ==1 else pcnt
        #cnu+=1
    
    y_true.append(r[1])
    y_pred.append(predSen)
    print(r[1],":", "scores:", VaSa.polarity_scores(r[0]))
    print(r[1],":", is_positive(r[0]),is_neutral(r[0]),is_negative(r[0]) )
    print(r[1],":", predSen)

print("1:", cp,"/",sum(testdata['sentiment']==1), "   c as -1 =",pcn, "  c as 0 =",pcnt )
print("0:", cnu,"/",sum(testdata['sentiment']==0), "   c as -1 =",ntcn, "  c as 1 =",ntcp )
print("-1:", cn,"/",sum(testdata['sentiment']==-1), "   c as 0 =",ncnt, "  c as 1 =",ncp )


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, labels=[1,0,-1],average='micro')

precision_recall_fscore_support(y_true, y_pred, labels=[1,0,-1])


from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred, labels=[1,0,-1], average='micro')
from sklearn.metrics import recall_score
recall = recall_score(y_true, y_pred, average='macro')

from sklearn.metrics import f1_score
f1_score(y_true, y_pred, average='macro')

from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_true, y_pred, labels=[1,0,-1], average='micro')

print('Precision: %.3f' % precision)

# $P = \frac{T_p}{T_p+F_p}$
# 
# $R = \frac{T_p}{T_p + F_n}$
# 
# $F1 = 2\frac{P \times R}{P+R}$

# Precision = TruePositives / (TruePositives + FalsePositives)
# Recall = TruePositives / (TruePositives + FalseNegatives)


# $Recall = Sum c in C TruePositives_c / Sum c in C (TruePositives_c + FalseNegatives_c)$
# 

# Without Neutrals 

testdata =allSentmntsWOneutral



def is_positive(rev: str) -> bool:
   
    n = VaSa.polarity_scores(rev)["neu"]
    c = VaSa.polarity_scores(rev)["compound"]
    return c>0.0 #c > 0 

def is_negative(rev: str) -> bool:
  
    return VaSa.polarity_scores(rev)["compound"] <=0.0 #< 0 

cp=0
cn=0
cnu=0
ncp=0
ntcp=0
pcn=0
ntcn =0
ncnt=0
pcnt=0
y_true=[]
y_pred=[]
for i,r in testdata.iterrows():
    
    if is_positive(r[0]) :
        predSen = 1
        cp = cp+1 if r[1]==1 else cp
        ncp=ncp+1 if r[1] ==-1 else ncp
        ntcp=ntcp+1 if r[1] ==0 else ntcp
    elif is_negative(r[0]) :
        predSen = -1
        cn=cn+1 if r[1] ==-1 else cn
        pcn=pcn+1 if r[1] ==1 else pcn
        ntcn = ntcn+1 if r[1] ==0 else ntcn
        #cn+=1
    
    y_true.append(r[1])
    y_pred.append(predSen)
    #print(r[1],":", "scores:", VaSa.polarity_scores(r[0]))
    #print(r[1],":", is_positive(r[0]),is_neutral(r[0]),is_negative(r[0]) )
    #print(r[1],":", predSen)

print("1:", cp,"/",sum(testdata['sentiment']==1), "   c as -1 =",pcn, "  c as 0 =",pcnt )
print("0:", cnu,"/",sum(testdata['sentiment']==0), "   c as -1 =",ntcn, "  c as 1 =",ntcp )
print("-1:", cn,"/",sum(testdata['sentiment']==-1), "   c as 0 =",ncnt, "  c as 1 =",ncp )



precision_recall_fscore_support(y_true, y_pred, labels=[1,-1],average='micro')


precision_recall_fscore_support(y_true, y_pred, labels=[1,-1],average='weighted') #weighted

