import os
import numpy
import pandas
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
    X, _ = load_dataset("{}/img".format(DATA_PATH))
    test_X = numpy.array(X)
    path = "model.pkl"
    model = autokeras.utils.pickle_from_file(path)
    vec_X = model.predict(test_X[0:2], output_index=-2)
    df = pandas.DataFrame(vec_X)
    df.to_csv("vec.csv", index=False)