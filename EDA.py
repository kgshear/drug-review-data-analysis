import nltk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
# from nltk.corpus import wordnet
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
import string
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sb
from transformers import pipeline
from sklearn.model_selection import train_test_split
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import hdbscan


class eda:
    data = None
    stopwords = stopwords.words('english')
    target_feature = "rating"
    train_filepath = "data/processed/train_data_processed.csv"
    test_filepath = "data/processed/test_data_processed.csv"
    train_filepath_filtered = "data/filtered/train_data_filtered_rating.csv"
    test_filepath_filtered = "data/filtered/test_data_filtered_rating.csv"
    small_filepath = "data/test/small_file_test_processed.csv"
    model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, truncation=True,
                              max_length=512,   batch_size=4)

    def __init__(self):
        # not_stopwords = ["me", "not", "didn't", "no", "wouldn't", "couldn't", "wasn't", "shouldn't", "won't",
        #                  "aren't", "haven't", "isn't", "weren't"]
        # for word in not_stopwords:
        #     self.stopwords.remove(word)
        # self.get_raw_data()

        # self.clean_data(self.data)
        # self.feature_engineering(self.test_df, self.test_filepath)
        # self.clean_data(self.train_df)
        # self.feature_engineering(self.train_df, self.train_filepath)
        self.get_processed_data()
        # self.get_filtered_data()
        # self.correlation_pearson(self.train_df, self.target_feature)
        # self.covariance_matrix(self.train_df, self.target_feature)
        # self.balance_data(self.train_df, self.target_feature)
        x_train, x_test, y_train, y_test = train_test_split(self.data.drop(columns=[self.target_feature], inplace=False),
                                                            self.data[self.target_feature], test_size=0.2, random_state=2002)
        self.train_df = x_train.copy()
        self.train_df[self.target_feature] = y_train
        self.test_df = x_test.copy()
        self.test_df[self.target_feature] = y_test
#         self.test_sentiment_analysis(pd.concat([
#     self.test_df[["rating", "sentiment"]],
#     self.train_df[["rating", "sentiment"]]
# ], ignore_index=True))
        self.dimensionality_reduction(x_train, y_train)
        self.anomaly_detection(self.train_df)
        self.save_filtered_data()


    def get_raw_data(self):
        self.test_df = pd.read_csv('data/raw/drugsComTest_raw.tsv', sep='\t')
        self.train_df = pd.read_csv('data/raw/drugsComTrain_raw.tsv', sep='\t')
        # self.small_df = pd.read_csv('data/test/small_file_test_raw.tsv', sep='\t')

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)
        # self.small_df = pd.read_csv('data/test/small_file_test_processed.csv', sep=',')

    def get_filtered_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath_filtered, sep=',')
        self.train_df = pd.read_csv(self.train_filepath_filtered, sep=',')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)

    # def add_column(self):
    #     self.test_df["nlkt_sentiment"] = self.test_df["review"].apply(self.sentiment_analysis_nlkt)
    #     self.train_df["nlkt_sentiment"] = self.train_df["review"].apply(self.sentiment_analysis_nlkt)
    def save_filtered_data(self):
        self.test_df.to_csv(self.test_filepath_filtered, index=False)
        self.train_df.to_csv(self.train_filepath_filtered, index=False)

    def set_target(self, column_name):
        self.target_feature = column_name

    def clean_review(self, text):
        if text != text:
            # check if nan
            return text
        else:
            text = text.lower()
            punctuation = string.punctuation + '\n' + '\t'
            text = text.replace('039', '\'')
            words = text.split()
            # lemmatizer = WordNetLemmatizer()
            # words = [lemmatizer.lemmatize(word) for word in words]
            clean_text = " ".join([word for word in words if word.lower()])
            for char in punctuation:
                table = str.maketrans(" ", " ", char)
                clean_text = clean_text.translate(table)
            return clean_text

    def standardization(self, df):
        new_df = (df - df.mean()) / df.std()
        return new_df

    def clean_text(self, text):
        if text != text:
            # check if nan
            return text
        else:
            text = text.lower()
            punctuation = string.punctuation + '\n' + '\t'
            for char in punctuation:
                table = str.maketrans(" ", " ", char)
                text = text.translate(table)
            return text

    def sentiment_analysis_nlkt(self, text):
        text = str(text)
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        words = [lemmatizer.lemmatize(word) for word in text]
        words = " ".join([word for word in words])
        sent_analyzer = SentimentIntensityAnalyzer()
        scores = sent_analyzer.polarity_scores(words)
        return 1 if scores['pos'] > 0 else 0

    def sentiment_analysis_hf(self, df):
        dataset = Dataset.from_pandas(df)
        results = []
        label_dict = {"positive": 2, "neutral": 1, "negative": 0}
        for out in self.sentiment_task(KeyDataset(dataset, "review")):
            label = out["label"]
            results.append(label_dict[label])
        return results

    def test_sentiment_analysis(self, df):
        ratings = [df["rating"] < 4,
            (df["rating"] >= 4) & (df["rating"] <= 7),
            df["rating"] > 7]
        labels = [0, 1, 2]
        true_labels = np.select(ratings, labels)
        predicted_labels = df["sentiment"]
        cm = confusion_matrix(true_labels,predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        fig, ax = plt.subplots()
        sizes = pd.Series(predicted_labels).sort_values().value_counts(sort=False)
        print(sizes)
        ax.pie(sizes, labels=["Negative", "Neutral", "Positive"])
        plt.show()

    def clean_data(self, df):
        df.loc[df['condition'].str.contains("users found this comment helpful", na=False), 'condition'] = np.nan
        df['word_count'] = df["review"].apply(lambda x: len(str(x).split()))
        df["upper_count"] = df["review"].apply(lambda x: len([word for word in str(x).split() if word.isupper()]))
        df['review'] = df['review'].apply(self.clean_review)
        df.loc[:, ['condition', 'drugName']] = df.loc[:, ['condition', 'drugName']].applymap(
            self.clean_text).loc[:]
        na_rows = df[df.isna().any(axis=1)]
        print(f"There were {len(na_rows)} na rows before dealing with na conditions by matching drugs")
        drug_list = df["drugName"].dropna().unique()
        drug_match_df = df[df["drugName"].isin(drug_list)][["drugName", "condition"]].drop_duplicates(subset="drugName")
        drug_to_condition = drug_match_df.set_index("drugName")["condition"]
        na_rows = na_rows.copy()
        na_rows.loc[:, "condition"] = na_rows["drugName"].map(drug_to_condition)
        df.loc[na_rows.index] = na_rows
        remaining_na_rows = df[df.isna().any(axis=1)]
        print(f"There were {len(remaining_na_rows)} na rows after dealing with na conditions by matching drugs")
        na_rows = df[df.isna().any(axis=1)]
        print(f"There were {len(na_rows)} na rows before dealing with na conditions by matching text")
        condition_series = df["condition"].dropna(inplace=False).unique().tolist()
        condition_series.remove("eve")
        condition_series.remove("me")
        for condition in condition_series:
            pattern = r"[^a-z]" + re.escape(condition) + r"[^a-z|s]"
            na_rows.loc[na_rows['review'].str.contains(pattern), 'condition'] = condition
        df.loc[na_rows.index] = na_rows
        remaining_na_rows = df[df.isna().any(axis=1)]
        print(f"There were {len(remaining_na_rows)} na rows after dealing with na conditions by matching text")
        df.dropna(inplace=True)
        df.drop_duplicates(subset="review")

        print(f"There were {len(df)} rows before filtering out conditions and drugs that occur 10 times")
        df.groupby('condition').filter(lambda x : len(x)>1)
        df.groupby('drugName').filter(lambda x: len(x)>1)
        # df_condition = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
        # df_condition = pd.DataFrame(df_condition).reset_index()
        print(f"There were {len(df)} rows after filtering out conditions and drugs that occur 10 times")

    def feature_engineering(self, df, filepath):
        # encoding
        df["drugName"] = df["drugName"].astype('category')
        df["drugName"] = df["drugName"].cat.codes
        df["condition"] = df["condition"].astype('category')
        df["condition"] = df["condition"].cat.codes
        df["date"] = df["date"].astype('category')
        df["date"] = df["date"].cat.codes
        df.rename(columns={"Unnamed: 0": "id"}, inplace=True)
        df["word_count"] = self.standardization(df["word_count"])
        df["upper_count"] = self.standardization(df["upper_count"])
        df['average_useful'] = df.groupby('drugName')['usefulCount'].transform('mean')
        df['reviewQuality'] = (df['usefulCount'] > df['average_useful']).astype(int)
        # df['reviewQuality'] = temp_df.map(int)
        df = df.drop(columns=['average_useful'])
        df['usefulStandardized'] = self.standardization(df["usefulCount"])

        df["numberOfReviews"] = df.groupby('drugName')["drugName"].transform('count')
        avg_nor = df["numberOfReviews"].mean()
        df["reviewFrequency"] = (df["numberOfReviews"] > avg_nor).astype(int)
        df = df.drop(columns=['numberOfReviews'])

        df["sentiment"] = self.sentiment_analysis_hf(df)
        df["dummy_sentiment"] = df["sentiment"].copy()
        df = pd.get_dummies(df, columns=["dummy_sentiment"], drop_first=True, dtype='int') # reduce dimensionality
        df.to_csv(filepath, index=False)

    def dimensionality_reduction(self, x_train, y_train):
        # compare methods of reducing dimensionality
        features_to_remove = self.random_forest(x_train, y_train)
        # x_train.drop(columns=[*features_to_remove])
        self.train_df = self.train_df.drop(columns=[*features_to_remove])
        self.test_df = self.test_df.drop(columns=[*features_to_remove])
        # self.pca(x_train)
        # self.singular_value_decomp(df)
        # self.vif(x_train)

    def temp(self):
        self.test_df.drop("dummy_sentiment_0")

    def random_forest(self, dfx, dfy):
        print("Started random forest")
        x_train = dfx.drop(columns=["review", "id", "sentiment", "usefulCount"], inplace=False)
        y_train = dfy
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        features = x_train.columns
        importances = model.feature_importances_
        sorted_imp = np.argsort(importances)
        plt.title("Feature importances")
        plt.barh(range(len(sorted_imp)), importances[sorted_imp])
        plt.yticks(range(len(sorted_imp)), [features[i] for i in sorted_imp])
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

        threshold = 0.05
        print(importances)
        features_to_remove = []
        features_to_keep = []
        for i in sorted_imp:
            if importances[i] >= threshold:
                features_to_keep.append(features[i])
            else:
                features_to_remove.append(features[i])
        print("Remaining Features: ", features_to_keep)
        print("Eliminated Features: ", features_to_remove)
        return features_to_remove

    def pca(self, df):
        pca = PCA()
        df = df.drop(columns=["id", "review", "sentiment", "usefulCount"], inplace=False)
        pca.fit(MinMaxScaler().fit_transform(df))
        explained_variance_ratio = pca.explained_variance_ratio_
        cumvar = np.cumsum(explained_variance_ratio)
        n_com = np.argmax(cumvar >= 0.95) + 1
        n_com_95 = np.argmin(abs(cumvar - 0.95)) + 1

        print(f"Number of components needed to explain more than 95% of the variance: {n_com}")
        plt.plot(range(1, len(cumvar) + 1), cumvar)
        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.axvline(x=n_com_95, color='r', linestyle='-')
        plt.title('Variance vs. Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance')
        plt.grid(True)
        plt.show()

    def singular_value_decomp(self, df, column_name):
        df = df.drop(columns=[column_name, "review", "id", "sentiment", "usefulCount"], inplace=False)
        # y = df[column_name]
        vec = df.to_numpy()
        mean_list = np.mean(vec, axis=0)
        centered_vec = vec - mean_list
        u, s, vh = np.linalg.svd(centered_vec)
        singular_table = PrettyTable()
        singular_table.title = "Singular Values"
        singular_table.field_names = ["Values"]
        singular_table.add_row([f"{s[0]:.2f}"])
        singular_table.add_row([f"{s[1]:.2f}"])
        print(singular_table)

    def vif(self, df):
        df = df.drop(columns=["review", "id", "sentiment", "usefulCount"], inplace=False)
        vif_data = pd.DataFrame()
        vif_data["feature_name"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                           for i in range(len(df.columns))]
        plt.bar(vif_data["feature_name"], vif_data["VIF"])
        plt.title("VIF of Features in Dataset")
        plt.xlabel("Features")
        plt.ylabel("Variance Inflation Factor")
        plt.show()

    def covariance_matrix(self, df, column_name):
        # x = df.drop(columns=[column_name], inplace=False)
        # y = df[column_name]
        # data = pd.concat([x, y], axis=1)
        # print(df.dtypes)
        df = df.drop(columns=["review", "id", "sentiment", "usefulCount"], inplace=False)
        cov_matrix = df.cov()
        sb.heatmap(cov_matrix, cmap="YlGnBu", annot=True)
        plt.title("Covariance Matrix Heatmap")
        plt.show()

    def correlation_pearson(self, df, column_name):
        df = df.drop(columns=["review", "id", "sentiment", "usefulCount"], inplace=False)
        sb.heatmap(df.corr(method='pearson'), cmap="YlGnBu", annot=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def balance_data(self, df, column_name):
        value_list = ["drugName", "condition"]
        for item in value_list:
            print(f"Number of rows before removing unique {item}: ", len(self.data))
            value_counts = self.data[item].value_counts()
            self.data = self.data[self.data[item].isin(value_counts[value_counts > 5].index)]
            print(f"Number of rows after removing unique {item}: ", len(self.data))
        print(f"Number of rows before balancing: {len(df)}")
        print(df["condition"].value_counts())
        sm = SMOTE(random_state=2002)
        x_re, y_re = sm.fit_resample(df.drop(columns=[column_name], inplace=False), df[column_name])
        print(y_re.value_counts())
        df[column_name].value_counts().plot(kind='bar', color=['blue', 'orange'])
        plt.title('Imbalanced Dataset')
        plt.xlabel('Direction')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 5))
        pd.Series(y_re).value_counts().plot(kind='bar', color=['blue', 'orange'])
        plt.title('Balanced Dataset')
        plt.xlabel('Direction')
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        print(f"Number of rows after balancing: {len(x_re)}")

    def anomaly_detection(self, df):
        df = df.drop(columns=["review", "id", "usefulCount", "sentiment", self.target_feature])
        model = hdbscan.HDBSCAN(min_cluster_size=4)
        labels = model.fit_predict(MinMaxScaler().fit_transform(df))
        df["Cluster"] = labels
        filtered_df = self.train_df[df["Cluster"] != -1]

        print(f"Original shape: {df.shape}")

        self.train_df = filtered_df
        self.test_df = self.test_df.drop(columns=["review", "id", "usefulCount", "sentiment"])
        self.train_df = self.train_df.drop(columns=["review", "id", "usefulCount", "sentiment"])
        print(f"Filtered shape: {self.train_df.shape}")



if __name__ == "__main__":
    # nltk.download('omw-1.4')
    # nltk.download('wordnet')
    # nltk.download('vader_lexicon')
    obj = eda()
    # obj.random_forest("rating")

