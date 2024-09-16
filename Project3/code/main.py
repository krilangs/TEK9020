# -*- coding: utf-8 -*-
import os
import logging
import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch

from io import StringIO
from sklearn import neighbors, linear_model, svm, tree, ensemble
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, \
                            confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier


# Ignore warnings in output
import warnings
warnings.filterwarnings("ignore")

# Set default font size of text in plotting
matplotlib.rcParams['font.size'] = 14

# Log print output to file
logger = logging.getLogger(__name__)
logging.basicConfig(filename="ClusterResults.log", encoding="utf-8",
                    filemode="w", format="%(message)s", level=logging.INFO)
save_folder = "figures/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)



# Functions
def prepare_data():
    """
    Read IRIS dataset and sort into dataframes. Plot the original clusters.
    """
    IRIS = pd.read_csv("data/iris.csv")
    iris_data = IRIS.copy()
    iris_data = iris_data.drop("Id", axis=1)

    buf = StringIO()
    iris_data.info(buf=buf)
    logger.info(buf.getvalue())
    logger.info(iris_data["Species"].value_counts())

    with sns.plotting_context(rc={"axes.labelsize":18}):
        img = sns.pairplot(iris_data, hue="Species", size=3)
    img.fig.suptitle("Original IRIS cluster", y=1.02)
    plt.savefig(save_folder + "Original_class/Original cluster.png",
                bbox_inches="tight")

    iris_X = iris_data.iloc[:, [0, 1, 2, 3]]
    iris_Y = iris_data["Species"]

    return iris_data, iris_X, iris_Y


def visualize_clusters(X, Y, labels, title="", save=""):
    """
    Plot data clusters, with known or predicted classes.
    """
    plt.figure()
    plt.title(title)
    plt.scatter(X[Y == labels[0]]["SepalLengthCm"],
                X[Y == labels[0]]["SepalWidthCm"],
                s=50, c="#1f77b4", label="Iris-setosa")
    plt.scatter(X[Y == labels[1]]["SepalLengthCm"],
                X[Y == labels[1]]["SepalWidthCm"],
                s=50, c="#ff7f0e", label="Iris-versicolor")
    plt.scatter(X[Y == labels[2]]["SepalLengthCm"],
                X[Y == labels[2]]["SepalWidthCm"],
                s=50, c="#2ca02c", label="Iris-virginica")
    plt.legend(bbox_to_anchor=(0.63, 0.68), fontsize=12)
    plt.xlabel("SepalLengthCm")
    plt.ylabel("SepalWidthCm")
    plt.savefig(save_folder + save + title + ".png", bbox_inches="tight")
    plt.show()


def elbow_method(iris_X):
    """
    Plot Elbow method:
        Used to find optimal number of clusters to use for KMeans.
    """
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i,
                        init="k-means++",
                        max_iter=300,
                        n_init=10,
                        random_state=42)
        kmeans.fit(iris_X)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow method")
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.savefig(save_folder + "KMeans_class/Elbow.png", bbox_inches="tight")
    plt.show()


def Dendrogram(X):
    """
    Plot Dendrogram:
        Used to find the optimal number of clusters to use with the
        Agglomerative method.
    """
    Z = sch.linkage(X, method="median")
    plt.figure(figsize=(20,7))
    sch.dendrogram(Z)
    plt.title("Dendrogram")
    plt.ylabel("Euclidean distance")
    plt.xlabel("Variables")
    plt.savefig(save_folder + "Agglomerative_class/Dendrogram.png",
                bbox_inches="tight")
    plt.show()


def conf_mat(cm, title="", save=""):
    """
    Plot Confusion matrix.
    """
    plt.figure()
    labels = ["set", "vers", "virg"]
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp2.plot(cmap="YlGnBu")
    plt.title("Confusion Matrix " + title)
    plt.savefig(save_folder + save + title + "confmat.png",
                bbox_inches="tight")
    plt.show()


def Classification_clusters(iris_X, iris_Y, pred_model, title="", save=""):
    """
    Make and visualize the predicted clusters.
    """
    visualize_clusters(iris_X, pred_model, [1, 0, 2],
                       title + " IRIS predictions", save=save)

    pred_model = pd.DataFrame(pred_model, columns=["Predictions"])
    pred_model = pred_model.replace({1: "Iris-setosa",
                                     0: "Iris-versicolor",
                                     2: "Iris-virginica"})
    prediction_data = iris_X.copy()
    prediction_data["Predictions"] = pred_model

    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    conf_mat(confusion_matrix(iris_Y, pred_model, labels=labels),
             title=title, save=save)
    logger.info("Classification - " + title + " cluster:")
    logger.info(f"Accuracy Score: {accuracy_score(iris_Y, pred_model)}")
    logger.info("Classification report:")
    logger.info(classification_report(iris_Y, pred_model))

    plt.figure()
    img = sns.pairplot(prediction_data, hue="Predictions", size=3)
    img.fig.suptitle(title + " IRIS cluster", y=1.02)
    plt.savefig(save_folder + save + title + ".png", bbox_inches="tight")


def Classifier_scores(X, Y, y_true, model, cv=5, title="", save=""):
    """
    Plot the predicted clusters, confusion matrix and classification report
    of the classifier performances.
    """
    y_pred = cross_val_predict(model, X, Y, cv=cv)
    visualize_clusters(X, y_pred,
                       ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
                       title + " IRIS predictions", save=save)

    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    conf_mat(confusion_matrix(y_true, y_pred, labels=labels),
             title=title, save=save)
    logger.info("Classification - " + title + ":")
    logger.info(f"Accuracy Score:{accuracy_score(y_true, y_pred)}")
    logger.info("Classification report:")
    logger.info(classification_report(y_true, y_pred))


def Classifiers(X, Y, y_true, save=""):
    """
    Predict and visualize the clusters with different classifiers.
    """
    # KNeighborsClassifier
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    Classifier_scores(X, Y, y_true, knn, title="KNeighbors", save=save)

    # LogisticRegression
    LR = linear_model.LogisticRegression()
    Classifier_scores(X, Y, y_true, LR, title="LogisticRegression", save=save)

    # Support Vector Machine
    SVC = svm.SVC()
    Classifier_scores(X, Y, y_true, SVC, title="SVM", save=save)

    # Decision Tree
    DTC = tree.DecisionTreeClassifier(criterion="entropy")
    Classifier_scores(X, Y, y_true, DTC, title="Decision Tree", save=save)

    # Random Forest
    RF = ensemble.RandomForestClassifier(n_estimators=10, criterion="entropy")
    Classifier_scores(X, Y, y_true, RF, title="Random Forest", save=save)

    # Multi-layer perceptron
    MLP = MLPClassifier()
    Classifier_scores(X, Y, y_true, MLP, title="MLP", save=save)


#----------------------------------
if __name__=="__main__":
    logger.info("-----------------------------------------")
    logger.info("Original cluster:")
    iris_data, iris_X, iris_Y = prepare_data()
    # Plot original cluster
    subFolder = "Original_class/"
    visualize_clusters(iris_data, iris_data["Species"],
                       ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
                       title="IRIS original clustering",
                       save=subFolder)

    logger.info("Evaluate classifiers on original cluster:")
    Classifiers(iris_X, iris_Y, iris_Y, save=subFolder)
    logger.info("-----------------------------------------")

    ## Elbow to find number of clusters
    logger.info("Elbow and KMeans")
    subFolder = "KMeans_class/"
    elbow_method(iris_X)

    # Modelling with KMeans
    model_KMeans = KMeans(n_clusters=3,
                    init="k-means++",
                    max_iter=300,
                    n_init=10,
                    random_state=42)

    pred_kmeans = model_KMeans.fit_predict(iris_X)

    # Plot predicted clusters
    Classification_clusters(iris_X, iris_Y, pred_kmeans,
                            title="KMeans",
                            save=subFolder)

    pred_kmeans = pd.DataFrame(pred_kmeans, columns=["Predictions"])
    pred_kmeans = pred_kmeans.replace({1: "Iris-setosa",
                                       0: "Iris-versicolor",
                                       2: "Iris-virginica"})

    logger.info("Evaluate classifiers on KMeans cluster:")
    #Classifiers(iris_X, pred_kmeans, pred_kmeans, save=subFolder + "KMeans")
    Classifiers(iris_X, pred_kmeans, iris_Y, save=subFolder + "KMeans")

    logger.info("-----------------------------------------")


    # Hierachical cluster method
    subFolder = "Agglomerative_class/"
    logger.info("Dendrogram and Agglo:")

    # Dendrogram to find number of clusters
    Dendrogram(iris_X)

    model_Agglo = AgglomerativeClustering(n_clusters=3)
    pred_agglo = model_Agglo.fit_predict(iris_X)

    Classification_clusters(iris_X, iris_Y, pred_agglo,
                            title="Agglomerative",
                            save=subFolder)
    pred_agglo = pd.DataFrame(pred_agglo, columns=["Predictions"])
    pred_agglo = pred_agglo.replace({1: "Iris-setosa",
                                     0: "Iris-versicolor",
                                     2: "Iris-virginica"})

    logger.info("Evaluate classifiers on Agglo cluster:")
    #Classifiers(iris_X, pred_agglo, pred_agglo, save=subFolder + "Agglo")
    Classifiers(iris_X, pred_agglo, iris_Y, save=subFolder + "Agglo")

