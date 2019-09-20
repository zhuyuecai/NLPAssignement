import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

np.random.seed(500)


def getData(file_path):
    """
    @param: file_path, str, path to the data file
    @return: an np array with each element as a line in the file
    """
    with open(file_path, errors='replace') as f:
        lines = f.readlines()
        dd = np.array(lines)
    return dd


def getCorpus(neg, pos):
    corpus = {"text": [],
              "label": []}
    corpus["text"] = np.concatenate((neg, pos))
    corpus["label"] = np.array(
        ["-1" for i in range(len(neg))] + ["1" for i in range(len(pos))],
        dtype='str')
    return corpus


# preprocessing
def preprocess_entry(entry):
    entry = entry.lower()
    entry = word_tokenize(entry)
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag'
    # i.e if the word is Noun(N) or Verb(V) or something else.
    Final_words = [word_Lemmatized.lemmatize(word, tag_map[tag[0]])
                   for word, tag in pos_tag(entry)
                   if word not in stopwords.words('english')
                   and word.isalpha()]
    return str(Final_words)


def pipeline(corpus, test_size=0.3, vectorize='tfidf', classifier='NB'):
    vectorize_options = ('tfidf', 'count')
    cls_options = ('NB', 'SVM')
    output_cls = {'NB': "Naive Bayes",
                  'SVM': "SVM"}
    # split test and training set
    Train_X, Test_X, Train_Y, Test_Y = \
        model_selection.train_test_split(corpus['final_text'],
                                         corpus['label'],
                                         test_size=test_size)
    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)
    if vectorize == vectorize_options[0]:
        vectorizer = TfidfVectorizer(max_features=10000)
    vectorizer.fit(corpus['final_text'])
    Train_X_vec = vectorizer.transform(Train_X)
    Test_X_vec = vectorizer.transform(Test_X)
    # fit the training dataset on the NB classifier
    if classifier == cls_options[0]:
        cls = naive_bayes.MultinomialNB()
    elif classifier == cls_options[1]:
        cls = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    cls.fit(Train_X_vec, Train_Y)
    # predict the labels on validation dataset
    predictions = cls.predict(Test_X_vec)
    # Use accuracy_score function to get the accuracy
    print(output_cls[classifier],
          "Accuracy Score -> ",
          accuracy_score(predictions, Test_Y)*100)


if __name__ == "__main__":
    # load data
    neg = getData("rt-polaritydata/rt-polarity.neg")
    pos = getData("rt-polaritydata/rt-polarity.pos")
    corpus = getCorpus(neg, pos)
    print(corpus['text'].shape)
    print(corpus['label'].shape)
    # preprocessing
    corpus['final_text'] = [preprocess_entry(entry)
                            for entry in corpus['text']]
    pipeline(corpus, test_size=0.2, classifier='NB')
    pipeline(corpus, test_size=0.2, classifier='SVM')
