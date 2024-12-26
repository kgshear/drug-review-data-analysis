import gensim
import nltk
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
import string
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sb
from transformers import pipeline
from sklearn.model_selection import train_test_split
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import hdbscan


class eda:
    data = None
    stopwords = stopwords.words('english')
    target_feature = "rating"
    train_filepath = "data/processed/train_data_processed_2.csv"
    test_filepath = "data/processed/test_data_processed_2.csv"
    train_filepath_filtered = "data/filtered/train_data_filtered_useful.csv"
    test_filepath_filtered = "data/filtered/test_data_filtered_useful.csv"
    model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path, truncation=True,
                              max_length=512,   batch_size=4)

    def __init__(self):

        self.get_raw_data()
        self.clean_data(self.data)
        x_train, x_test, y_train, y_test = train_test_split(self.data.drop(columns=[self.target_feature], inplace=False),
                                                           self.data[self.target_feature], test_size=0.2,
                                                           random_state=2002)
        self.train_df = x_train.copy()
        self.train_df[self.target_feature] = y_train
        self.test_df = x_test.copy()
        self.test_df[self.target_feature] = y_test


        self.feature_engineering(self.test_df, self.test_filepath)
        self.feature_engineering(self.train_df, self.train_filepath)

        self.get_processed_data()

        self.test_df["usefulStandardized"] = self.standardization(self.test_df["usefulCount"])
        self.train_df["usefulStandardized"] = self.standardization(self.train_df["usefulCount"])
        self.test_df = self.test_df.drop(columns=["review", "id", "usefulCount", "sentiment", "reviewQuality"])
        self.train_df = self.train_df.drop(columns=["review", "id", "sentiment", "usefulCount", "reviewQuality"])

        self.train_df = self.anomaly_detection(self.train_df)
        self.test_df["rating"] = self.simplify_rating(self.test_df)
        self.train_df["rating"] = self.simplify_rating(self.train_df)
        self.train_df = self.train_df.drop(columns=["Cluster"], inplace=False)
        self.train_df = self.balance_data(self.train_df)

        self.dimensionality_reduction()
        self.correlation_pearson(self.train_df, self.target_feature)
        self.covariance_matrix(self.train_df, self.target_feature)
        self.save_filtered_data()

    def get_raw_data(self):
        self.test_df = pd.read_csv('data/raw/drugsComTest_raw.tsv', sep='\t')
        self.train_df = pd.read_csv('data/raw/drugsComTrain_raw.tsv', sep='\t')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)


    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.test_df =self.test_df.dropna()
        self.train_df = pd.read_csv(self.train_filepath, sep=',')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)

    def get_filtered_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath_filtered, sep=',')
        self.train_df = pd.read_csv(self.train_filepath_filtered, sep=',')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)

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
    def simplify_rating(self, df):
        ratings = [df["rating"] < 6,
                   df["rating"] > 5]
        labels = [0, 1]
        true_labels = np.select(ratings, labels)
        return true_labels
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
        # df_condition = df.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
        # df_condition = pd.DataFrame(df_condition).reset_index()

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

    def dimensionality_reduction(self):
        # compare methods of reducing dimensionality vif, random forest, anamoly detection, pca, covar, corr, resample

        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        y_train = self.train_df[self.target_feature]
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        y_test = self.test_df[self.target_feature]
        # x_train, x_test = self.pca(x_train, x_test)
        vif_features_to_remove = self.vif(x_train)
        rf_features_to_remove = self.random_forest(x_train, y_train)  # we don't want to remove everything
        x_train[self.target_feature] = y_train
        self.train_df = x_train
        x_test[self.target_feature] = y_test
        self.test_df = x_test
        if len(rf_features_to_remove) > 0:
            self.train_df = self.train_df.drop(columns=[*rf_features_to_remove])
            self.test_df = self.test_df.drop(columns=[*rf_features_to_remove])


    def vectorize_text(self):
        # bag of words ish
        reviews = self.data["review"].apply(lambda sentence: sentence.split())
        model = gensim.models.Word2Vec(sentences=reviews, min_count=1, vector_size=25, window=5)
        wv = model.wv

        train_reviews = self.train_df["review"].apply(lambda sentence: sentence.split())
        test_reviews = self.test_df["review"].apply(lambda sentence: sentence.split())
        train_embeddings = train_reviews.apply(lambda x: np.mean([wv[word] for word in x if word in wv], axis=0))
        train_embedding_df = pd.DataFrame(
            train_embeddings.to_list(), columns=[f"emb{i+1}" for i in range(model.vector_size)])
        test_embeddings = test_reviews.apply(lambda x: np.mean([wv[word] for word in x if word in wv], axis=0))
        test_embedding_df = pd.DataFrame(
            test_embeddings.to_list(), columns=[f"emb{i + 1}" for i in range(model.vector_size)])
        return train_embedding_df, test_embedding_df

    def random_forest(self, dfx, dfy):
        print("Started random forest")
        x_train = dfx
        y_train = dfy
        print(dfy.head())
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

    def pca(self, train, test):
        useful_test = test["usefulStandardized"]
        useful_train = train["usefulStandardized"]
        test.drop(columns=["usefulStandardized"])
        train.drop(columns=["usefulStandardized"])
        scaler = StandardScaler()
        scaler.fit(train)
        train_scaled = scaler.transform(train)
        test_scaled = scaler.transform(test)
        pca = PCA(n_components=0.95)
        pca.fit(train_scaled)
        train_pca = pca.transform(train_scaled)
        test_pca = pca.transform(test_scaled)
        col_names = [f"PCA{i + 1}" for i in range(train_pca.shape[1])]
        train_df = pd.DataFrame(train_pca, columns=col_names, index=train.index)
        test_df = pd.DataFrame(test_pca, columns=col_names, index=test.index)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumvar = np.cumsum(explained_variance_ratio)
        n_com = np.argmax(cumvar >= 0.95) + 1
        n_com_95 = np.argmin(abs(cumvar - 0.95)) + 1

        print(f"Number of components needed to explain more than 95% of the variance: {n_com}")
        cov_matrix = np.cov(train_scaled, rowvar=False)
        condition_num = np.linalg.cond(cov_matrix)
        print(f"Condition num: {condition_num}")

        plt.plot(range(1, len(cumvar) + 1), cumvar)
        plt.axhline(y=0.95, color='r', linestyle='-')
        plt.axvline(x=n_com_95, color='r', linestyle='-')
        plt.title('Variance vs. Number of Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Variance')
        plt.grid(True)
        plt.show()
        #
        train_df["usefulStandardized"] = useful_train
        test_df["usefulStandardized"] = useful_test
        return train_df, test_df

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
        threshold = 5
        vif_data = pd.DataFrame()
        vif_data["feature_name"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                           for i in range(len(df.columns))]
        features_to_elim = vif_data[vif_data["VIF"] > threshold]["feature_name"]
        plt.bar(vif_data["feature_name"], vif_data["VIF"])
        plt.title("VIF of Features in Dataset")
        plt.xlabel("Features")
        plt.ylabel("Variance Inflation Factor")
        plt.show()
        features_to_keep = df.drop(columns=[*features_to_elim], inplace=False).columns
        print("Remaining Features: ", features_to_keep)
        print("Eliminated Features: ", features_to_elim)
        return features_to_elim

    def covariance_matrix(self, df, column_name):

        df = df.drop(columns=[self.target_feature], inplace=False)
        cov_matrix = df.cov()
        sb.heatmap(cov_matrix, cmap="YlGnBu", annot=True)
        plt.title("Covariance Matrix Heatmap")
        plt.show()

    def correlation_pearson(self, df, column_name):
        df = df.drop(columns=[self.target_feature], inplace=False)
        sb.heatmap(df.corr(method='pearson'), cmap="YlGnBu", annot=True)
        plt.title("Correlation Matrix Heatmap")
        plt.show()

    def balance_data(self, df):
        conditions = df["rating"].value_counts().sort_values(ascending=False)
        conditions.plot(kind='bar')
        plt.title('Rating Before Balancing')
        plt.xlabel("Rating")
        plt.ylabel("Count")
        plt.grid(True)

        oversampler = SMOTE(k_neighbors=2)
        over_x, over_y = oversampler.fit_resample(df.drop(columns=[self.target_feature], inplace=False), df[self.target_feature])
        over_x[self.target_feature] = over_y

        print("over x value counts", over_x[self.target_feature].value_counts())
        print("Number of rows after resampling: ", len(over_x))

        plt.figure(figsize=(10, 5))
        pd.Series(over_x[self.target_feature]).value_counts().plot(kind='bar')
        plt.title('Rating After Balancing')
        plt.xlabel('Rating')
        # plt.xticks(rotation=35)
        plt.ylabel('Count')
        plt.grid(True)
        plt.show()
        return over_x

    def anomaly_detection(self, df):
        model = hdbscan.HDBSCAN(min_cluster_size=4)
        labels = model.fit_predict(MinMaxScaler().fit_transform(df))
        df["Cluster"] = labels
        filtered_df = self.train_df[df["Cluster"] != -1]
        print(f"Original shape: {df.shape}")

        print(f"Filtered shape: {filtered_df.shape}")
        return filtered_df


if __name__ == "__main__":
    nltk.download('omw-1.4')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    obj = eda()

