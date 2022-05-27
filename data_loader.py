import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def data_loader( plot = False):
    train = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_train.csv')
    y_train, X_train_0, X_train_1 = np.array(train["y_train"]), train["feature1"], train["feature2"]
    X_train = np.array([X_train_0, X_train_1]).T

    test = pd.read_csv('./quantum_circuit_designed_dataset/TEST_phi1_repeat1_test.csv')
    y_test, X_test_0, X_test_1 = np.array(test["y_test"]), test["feature1"], test["feature2"]
    X_test = np.array([X_test_0, X_test_1]).T

    if plot==True:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.scatter(X_train.T[0], X_train.T[1], c=y_train,cmap=plt.cm.coolwarm, label='Train')
        ax.scatter(X_test.T[0], X_test.T[1], c=y_test, marker='x',cmap=plt.cm.coolwarm, label='Test')
        legend_elements = [Line2D([0], [0], marker='x', color='black', label='Test', markerfacecolor='black', lw=0),
                           Line2D([0], [0], marker='o', color='black', label='Train', markerfacecolor='black', lw=0),
                           ]
        ax.legend(handles=legend_elements, loc="lower right")
        plt.show()

    # print(X_train)
    # print(y_train)
    return (X_train, y_train, X_test, y_test)


data_loader()
