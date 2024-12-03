import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np

from classifier_evaluator import ClassifierEvaluator


class ClusteringAssociationAnalysis:
    train_filepath = "data/filtered/train_data_filtered.csv"
    test_filepath = "data/filtered/test_data_filtered.csv"
    target_feature = "condition"
    test_df = None
    train_df = None
    def __init__(self):
        self.get_processed_data()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')

    def kmeans(self):
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        bestcluster = None
        cluster_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        cluster_var = []
        for cluster_size in cluster_sizes:
            kmeans = KMeans(n_clusters=cluster_size, init='k-means++')
            kmeans.fit(x_train)
            sc = silhouette_score(x_train, list(kmeans.labels_))
            cluster_var.append(kmeans.inertia_)
            if bestcluster is None or sc > bestcluster[1]:
                bestcluster = (cluster_size, sc, kmeans.labels_)
        plt.plot(cluster_sizes, cluster_var)
        plt.title("Cluster Variation for Various Cluster Sizes")
        plt.xlabel("Cluster Size")
        plt.ylabel("Within-Cluster Variation")
        plt.show()
        kmeans = KMeans(n_clusters=bestcluster[0], init='k-means++')
        kmeans.fit(x_train)
        y_pred = kmeans.fit_predict(x_test)
        obj = ClassifierEvaluator()
        obj.display_confusion_matrix(y_pred,y_test, "KMeans")
        obj.display_result(y_pred,y_test, "KMeans")

    def dbscan(self):
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        db = DBSCAN()
        db.fit(x_train)
        y_pred = db.fit_predict(x_test)
        obj = ClassifierEvaluator()
        obj.display_confusion_matrix(y_pred, y_test, "DBScan")
        obj.display_result(y_pred, y_test, "DBScan")

    def apriori(self):
        binary_df = pd.get_dummies(self.train_df[['ratingLevel', 'usefulLevel', 'drugName', 'condition']])
        ap_model = apriori(binary_df, use_colnames=True)
        a_rules = association_rules(ap_model)
        print(a_rules)

