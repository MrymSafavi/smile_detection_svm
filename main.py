import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from feature_extraction import Extractor

# Path of genki4 and lables
DIR_PATH = "genki4/files"
LABEL_DIR_PATH = "genki4/labels.txt"


# 2162 smile
# 1838 not smile

# show all data with their classification
def show_data(data, labels):
    tsne = TSNE(n_components=2, perplexity=9)
    tsne_data = tsne.fit_transform(data)
    plt.scatter(tsne_data[:, 0],
                tsne_data[:, 1],
                c=labels)
    plt.title(f'The Classification of data')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def extract_labels():
    counter = 0
    with open(LABEL_DIR_PATH, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        counter += 1
        labels.append(int(line.split()[0]))
    return labels


def make_data():
    labels = extract_labels()

    extractor = Extractor()
    extractor.apply_extracting(DIR_PATH)
    features_extracted = extractor.features_extracted

    datas = np.array(features_extracted)
    labels = np.array(labels)

    # show_data(datas, labels)

    return features_extracted, labels


def train(X_train, y_train):
    svm = SVC(kernel='rbf', C=10)
    svm.fit(X_train, y_train)
    return svm


def test(svm, X_test, y_test):
    y_pred = svm.predict(X_test)
    print(y_pred)
    print(y_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_svm(svm):
    with open('detect_smile_model-final.pkl', 'wb') as f:
        pickle.dump(svm, f)


def load_svm():
    with open('detect_smile_model1.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# def test_on_clip():
#     use_model = UseModelOnClip()
#     use_model.start()
#     use_model.end()


def main():
    print('Extracting features...')
    features_extracted, labels = make_data()
    print('Splitting train and test data...')
    X_train, X_test, y_train, y_test = train_test_split(features_extracted, labels, test_size=0.3, random_state=42)
    print('Training...')
    svm = train(X_train, y_train)
    print('Testing...')
    accuracy = test(svm, X_test, y_test)
    print('Accuracy:', accuracy)
    save_svm(svm)


if __name__ == '__main__':
    main()
