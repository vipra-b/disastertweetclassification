
import pandas as pd
from sklearn.model_selection import train_test_split

from Process import Process


def main():
    base = Process()

    train_path = open('/Users/viprabindal/Downloads/nlp-getting-started/train.csv')
    total = pd.read_csv(train_path)
    total.drop(labels=["id", "keyword"], axis = 1)
    tweets = total["text"]
    tweets = base.preprocess(tweets)
    labels = total["target"]

    x_train, x_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.2, random_state = 1)

    preds = base.pred(x_train, y_train, x_test)

    print(base.accuracy(preds, y_test))



if __name__ == "__main__":
    main()

