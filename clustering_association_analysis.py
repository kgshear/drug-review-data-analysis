import mlxtend
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score, \
    mutual_info_score
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import re
from sklearn.preprocessing import LabelBinarizer



class ClusteringAssociationAnalysis:
    train_filepath = "data/filtered/train_data_filtered_useful.csv"
    test_filepath = "data/filtered/test_data_filtered_useful.csv"
    target_feature = "condition"
    test_df = None
    train_df = None
    drug_df = pd.DataFrame()
    condition_df = pd.DataFrame()
    def __init__(self):
        self.get_processed_data()
        self.get_categorical_dfs()
        self.run_all_functions()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')

    def get_categorical_dfs(self):
        raw_test_df = pd.read_csv('data/raw/drugsComTest_raw.tsv', sep='\t')
        raw_train_df = pd.read_csv('data/raw/drugsComTrain_raw.tsv', sep='\t')
        df = pd.concat([raw_test_df, raw_train_df], ignore_index=True)
        df.loc[df['condition'].str.contains("users found this comment helpful", na=False), 'condition'] = np.nan

        na_rows = df[df.isna().any(axis=1)]
        drug_list = df["drugName"].dropna().unique()
        drug_match_df = df[df["drugName"].isin(drug_list)][["drugName", "condition"]].drop_duplicates(subset="drugName")
        drug_to_condition = drug_match_df.set_index("drugName")["condition"]
        na_rows = na_rows.copy()
        na_rows.loc[:, "condition"] = na_rows["drugName"].map(drug_to_condition)
        df.loc[na_rows.index] = na_rows
        na_rows = df[df.isna().any(axis=1)]
        condition_series = df["condition"].dropna(inplace=False).unique().tolist()
        condition_series.remove("eve")
        condition_series.remove("me")
        for condition in condition_series:
            pattern = r"[^a-z]" + re.escape(condition) + r"[^a-z|s]"
            na_rows.loc[na_rows['review'].str.contains(pattern), 'condition'] = condition
        df.loc[na_rows.index] = na_rows
        df.dropna(inplace=True)


        self.drug_df["d_name"] = df["drugName"]
        self.drug_df["drugName"] = df["drugName"].astype('category').cat.codes

        self.condition_df["c_name"] = df["condition"]
        self.condition_df["condition"] = df["condition"].astype('category').cat.codes
        self.drug_df = self.drug_df.drop_duplicates()
        self.condition_df = self.condition_df.drop_duplicates()

    def run_all_functions(self):
        self.kmeans()
        self.dbscan()
        self.apriori()

    def kmeans(self):

        bestcluster = None
        cluster_sizes = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        cluster_var = []
        for cluster_size in cluster_sizes:
            kmeans = KMeans(n_clusters=cluster_size, init='k-means++')
            kmeans.fit(self.train_df)
            sc = silhouette_score(self.train_df, list(kmeans.labels_))
            cluster_var.append(kmeans.inertia_)
            if bestcluster is None or sc > bestcluster[1]:
                bestcluster = (cluster_size, sc, kmeans.labels_)
        print("Best cluster size", bestcluster[0])
        plt.plot(cluster_sizes, cluster_var)
        plt.title("Cluster Variation for Various Cluster Sizes")
        plt.xlabel("Cluster Size")
        plt.ylabel("Within-Cluster Variation")
        plt.show()
        kmeans = KMeans(n_clusters=4, init='k-means++')
        kmeans.fit(self.train_df)

        silhouette = silhouette_score(self.train_df, kmeans.labels_)
        db_index = davies_bouldin_score(self.train_df, kmeans.labels_)
        ch_index = calinski_harabasz_score(self.train_df, kmeans.labels_)
        ari = adjusted_rand_score(self.train_df[self.target_feature], kmeans.labels_)
        mi = mutual_info_score(self.train_df[self.target_feature], kmeans.labels_)

        print("Silhouette Score: ", silhouette)
        print("DB Index: ", db_index)
        print("Calinski-h: ",ch_index)
        print("adj rand: ", ari)
        print("Mutual info: ", mi)

    def dbscan(self):
        print("DBSCAN")
        y_train = self.train_df[self.target_feature]

        db = DBSCAN()
        db.fit(self.train_df)
        labels = db.labels_
        noise_count = list(labels).count(-1)
        num_clus = len(set(labels)) - (1 if noise_count > 0 else 0)

        silhouette = silhouette_score(self.train_df, labels)
        db_index = davies_bouldin_score(self.train_df, labels)
        ch_index = calinski_harabasz_score(self.train_df, labels)
        ari = adjusted_rand_score(y_train, labels)
        mi = mutual_info_score(y_train, labels)

        print("Silhouette Score: ", silhouette)
        print("DB Index: ", db_index)
        print("Calinski-h: ", ch_index)
        print("adj rand: ", ari)
        print("Mutual info: ", mi)
        print("DBSCAN number of clusters: ", num_clus)


    def apriori(self):
        self.train_df["drugName"] = self.train_df["drugName"].map(self.drug_df.set_index("drugName")["d_name"])
        self.train_df["condition"] = self.train_df["condition"].map(self.condition_df.set_index("condition")["c_name"])
        binarizer = LabelBinarizer()

        drugs = binarizer.fit_transform(self.train_df['drugName'])
        drugs_df = pd.DataFrame(drugs, columns=binarizer.classes_)
        conditions = binarizer.fit_transform(self.train_df['condition'])
        conditions_df = pd.DataFrame(conditions, columns=binarizer.classes_)

        df = pd.concat([drugs_df, conditions_df], axis=1)
        df["rating"] = self.train_df["rating"]
        make_bool = {False: 0, True: 1}
        df = df.replace(make_bool)
        ap_model = apriori(df, min_support=0.005, use_colnames=True, verbose=1)
        print("Model: ", ap_model)
        a_rules = association_rules(ap_model, metric="confidence", min_threshold=0.8, num_itemsets=len(df))
        sorted_rules = a_rules.sort_values(["confidence", "lift"], ascending=[False, False])
        print(sorted_rules.to_string())

if __name__ == "__main__":
    obj = ClusteringAssociationAnalysis()
