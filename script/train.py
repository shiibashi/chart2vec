import os
import numpy
import pandas
import datetime
import autokeras

DIR_PATH = os.path.dirname(__file__) or "."
#DIR_PATH = "."
DATA_PATH = "{}/../data".format(DIR_PATH)

from PIL import Image

LABEL_MAP = {
    "label1": 0,
    "label2": 1
}

def load_dataset(dirname):
    X = []
    Y = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        label = filename.split("_")[1].split(".")[0]
        label_id = LABEL_MAP[label]
        img = Image.open("{}/{}".format(dirname, filename))
        arr = numpy.array(img)
        X.append(arr)
        Y.append(label_id)
    return X, Y

if __name__ == "__main__":
    X, Y = load_dataset("{}/img".format(DATA_PATH))
    train_rate = 0.8

    train_X = numpy.array(X[0:int(len(X)*train_rate)])
    train_Y = numpy.array(Y[0:int(len(Y)*train_rate)])
    test_X = numpy.array(X[int(len(X)*train_rate):])
    test_Y = numpy.array(Y[int(len(Y)*train_rate):])
    
    print("train_X_shape: {}".format(train_X.shape))
    print("train_Y_shape: {}".format(train_Y.shape))
    print("test_X_shape: {}".format(test_X.shape))
    print("test_Y_shape: {}".format(test_Y.shape))

    model = autokeras.ImageClassifier()
    model.fit(train_X, train_Y, time_limit=0.5*60*60)
    print(model.cnn.searcher.history)
    model.final_fit(
        train_X,
        train_Y,
        test_X,
        test_Y,
        retrain=False)
    print(model.cnn.best_model.produce_model())
    path = "model.pkl"
    model.export_autokeras_model(path)