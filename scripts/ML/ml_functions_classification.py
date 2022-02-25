import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split # Used to split dataset
from sklearn.preprocessing import StandardScaler # Used in feature scaling

from sklearn.linear_model import LogisticRegression # Used for Logistic Regression Classification
from sklearn.neighbors import KNeighborsClassifier # Used for KNN Classification
from sklearn.svm import SVC # Used for SVM Classification
from sklearn.naive_bayes import GaussianNB # Used for Naive Bayes Classification
from sklearn.tree import DecisionTreeClassifier # Used for Decision Tree Classification
from sklearn.ensemble import RandomForestClassifier # Used for Random Forest Classification

from sklearn.metrics import confusion_matrix # Used for Confusion Matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, hamming_loss, jaccard_score, matthews_corrcoef # Different metrics
from sklearn.metrics import precision_score, recall_score, zero_one_loss # Different metrics
from matplotlib.colors import ListedColormap # Used for Visualization

def data(dataset):
    dataset = pd.read_csv(dataset)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return X, y

# SPLITTING DATASET INTO TRAINING AND TESTING SET
def training(X, y, testSize = 0.2, randomState = 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testSize, random_state = randomState)

    return X_train, X_test, y_train, y_test

# FEATURE SCALING
def scaling(X_train, X_test, i = 0):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test


############################
# TRAINING LOGISTIC REGRESSION MODEL ON TRAINING SET
def logistic_regression_classification(X_train, y_train, randomState = 0):
    classifier = LogisticRegression(random_state = randomState)
    classifier.fit(X_train, y_train)
    return classifier

# TRAINING K NEAREST NEIGHBORS MODEL ON TRAINING SET
def knn_classification(X_train, y_train, nNeighbors = 5, metric_ = '', p_ = 2):
    classifier = KNeighborsClassifier(n_neighbors = nNeighbors, metric = metric_, p = p_)
    classifier.fit(X_train, y_train)
    return classifier

def svm_classification(X_train, y_train, kernel_ = 'linear', randomState = 0):
    classifier = SVC(kernel = kernel_, random_state = randomState)
    classifier.fit(X_train, y_train)
    return classifier

def kernel_svm_classification(X_train, y_train, kernel_ = 'rbf', randomState = 0):
    classifier = SVC(kernel = kernel_, random_state = randomState)
    classifier.fit(X_train, y_train)
    return classifier

def naive_bayes_classification(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

def decision_tree_classification(X_train, y_train, criterion_ = 'entropy', randomState = 0):
    classifier = DecisionTreeClassifier(criterion = criterion_, random_state = randomState)
    classifier.fit(X_train, y_train)
    return classifier

def random_forest_classification(X_train, y_train, nEstimators = 10, criterion_ = 'entropy', randomState = 0):
    classifier = RandomForestClassifier(n_estimators = nEstimators, criterion = criterion_, random_state = randomState)
    classifier.fit(X_train, y_train)
    return classifier

############################
# PREDICT LOGISTIC REGRESSION CLASSIFIER
# vector = [DATA POINTS e.g. 30,87000]
def predict_lrc(classifier, sc, vector):
    y_pred = classifier.predict(sc.transform([vector]))

    return y_pred


############################
# PREDICT LOGISTIC REGRESSION CLASSIFIER via TEST RESULTS
def predict_lrc_test(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    return y_pred


############################
# METRICS
def confusion_matrix_(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    return cm

def accuracy_score_(y_test, y_pred):
    return accuracy_score(y_test, y_pred)

def balanced_accuracy_score_(y_test, y_pred):
    return balanced_accuracy_score(y_test, y_pred)

def f1_score_(y_test, y_pred, average_ = None):
    # average_ = binary, micro, macro, weighted, samples, None (all in a vector)
    return f1_score(y_test, y_pred, average = average_)

def hamming_loss_(y_test, y_pred):
    return hamming_loss(y_true, y_pred)

def jaccard_score_()y_test, y_pred, average_ = None):
    # average_ = binary, micro, macro, weighted, samples, None (all in a vector)
    return jaccard_score(y_true, y_pred, average = average_)

def matthews_corrcoef_(y_test, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def precision_score_()y_test, y_pred, average_ = None):
    # average_ = binary, micro, macro, weighted, samples, None (all in a vector)
    return precision_score(y_true, y_pred, average = average_)

def recall_score_()y_test, y_pred, average_ = None):
    # average_ = binary, micro, macro, weighted, samples, None (all in a vector)
    return recall_score(y_true, y_pred, average = average_)

def zero_one_loss_(y_test, y_pred, normalize_ = False):
    return zero_one_loss(y_true, y_pred, normalize = normalize_)

############################
# VISUALIZATION

# LOGISTIC REGRESSION CLASSIFIER
def graph_lrc(X_train, y_train, title = 'Title', xlabel = 'xLabel', ylabel = 'yLabel', color_s = 'red', color_l = 'green'):
    X_set, y_set = sc.inverse_transform(X_train), y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                         np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha = 0.75, cmap = ListedColormap((color_s, color_l)))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap((color_s, color_l))(i), label = j)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
