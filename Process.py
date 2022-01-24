
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

class Process:

    # constructor
    def __init__(self):
        self = self

    def preprocess(self, data):
        # cleaning the data
        data = data.str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
        data = data.str.replace(r"[\"\'\|\?\=\.\@\#\*\,]", '')


        #create the transformer object
        bag_transform = CountVectorizer(analyzer = 'word').fit(data)
        #print(bag_transform.vocabulary_)

        #use the transformer to make bag of word vectors
        data = bag_transform.transform(data)
        return data

    def pred(self, x_train, y_train, x_test):
        nb_model = MultinomialNB()
        nb_model.fit(x_train,y_train)
        pred = nb_model.predict(x_test)
        return pred

    def accuracy(self, pred, y_test):
        cm = confusion_matrix(y_test, pred)
        accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
        return accuracy
