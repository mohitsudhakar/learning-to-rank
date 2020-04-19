import numpy as np
import LambdaRankNN
from LambdaRankNN import RankNetNN

from dataloader import load_data

if __name__ == '__main__':

    folder = '/Users/mohit/Documents/Grad_Courses/spring20/info-retrieval/learning_to_rank/MQ2008/Fold1/'
    X, y, qid = load_data(folder + 'train.txt')
    Xtest, ytest, qid_test = load_data(folder + 'test.txt')

    epochs = 20
    # train model
    ranker = RankNetNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu'), solver='adam')
    # ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(32,16,8,), activation=('relu', 'relu', 'relu'), solver='adam')
    ranker.fit(X, y, qid, epochs=epochs)
    y_pred = ranker.predict(Xtest)
    print("Train")
    ranker.evaluate(X, y, qid, eval_at=5)
    ranker.evaluate(X, y, qid, eval_at=10)
    ranker.evaluate(X, y, qid, eval_at=50)
    ranker.evaluate(X, y, qid, eval_at=100)
    print("Test")
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=5)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=10)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=50)
    ranker.evaluate(Xtest, ytest, qid_test, eval_at=100)

