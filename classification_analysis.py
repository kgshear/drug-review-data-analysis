
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from classifier_evaluator import ClassifierEvaluator


class ClassificationAnalysis:
    train_filepath = "data/filtered/train_data_filtered_useful.csv"
    test_filepath = "data/filtered/test_data_filtered_useful.csv"
    target_feature = "rating"
    test_df = None
    train_df = None
    def __init__(self):
        self.get_processed_data()
        self.run_classifiers()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')
        self.data = pd.concat([self.test_df, self.train_df], ignore_index=True)

    def run_classifiers(self):
        self.decision_tree_preprune()
        self.decision_tree_postprune()
        self.logistic_regression()
        self.knn()
        self.svm()
        self.naive_bayes()
        self.neural_network()

    def decision_tree_preprune(self):
        print("Pre prune DT")
        param_grid = [{'max_depth': [1, 2, 3, 4, 5, 10, 15, 20],
                       'min_samples_split': [20, 30, 40],
                       'min_samples_leaf': [10, 20, 30],
                       'criterion': ['gini', 'entropy', 'log_loss'],
                       'splitter': ['best', 'random'],
                       'max_features': ['sqrt', 'log2']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        model = DecisionTreeClassifier(random_state=2002)
        dt_cv = GridSearchCV(model, param_grid, cv=5)
        dt_cv.fit(x_train, y_train)
        print("Tuned DT Parameters: {}".format(
            dt_cv.best_params_))

        best_model = DecisionTreeClassifier(random_state=5805, **dt_cv.best_params_)
        best_model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, best_model.predict(x_train))
        test_accuracy = accuracy_score(y_test, best_model.predict(x_test))

        print(f"The train accuracy is {train_accuracy:.2f}")
        print(f"The test accuracy is {test_accuracy:.2f}")

        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred,y_test,y_pred_prob,"Pre-Pruned Decision Tree")

    def decision_tree_postprune(self):
        print("Post prune DT")

        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        model = DecisionTreeClassifier(random_state=2002)
        path = model.cost_complexity_pruning_path(x_train, y_train)
        alphas = path.ccp_alphas
        param_grid = [{'criterion': ['gini', 'entropy', 'log_loss'],
                       'splitter': ['best', 'random'],
                       'max_features': ['sqrt', 'log2'],
                       'ccp_alpha': alphas}]
        dt_cv = GridSearchCV(model, param_grid, cv=5)
        dt_cv.fit(x_train, y_train)
        print("Tuned DT Parameters: {}".format(
            dt_cv.best_params_))



        best_model = DecisionTreeClassifier(random_state=5805, **dt_cv.best_params_)
        best_model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, best_model.predict(x_train))
        test_accuracy = accuracy_score(y_test, best_model.predict(x_test))

        print(f"The train accuracy is {train_accuracy:.2f}")
        print(f"The test accuracy is {test_accuracy:.2f}")

        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred,y_test,y_pred_prob,"Post-Pruned Decision Tree")

    def logistic_regression(self):
        print("Logistic Regression")
        param_grid = [{'max_iter': [50, 60, 70, 80, 90, 100, 200],
                       'C': [0.01,0.1,1,10,100, 1000],
                       'solver': ['lbfgs','liblinear', 'sag', 'saga', 'newton-cholesky']}]
        scaler = StandardScaler()
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = scaler.fit_transform(self.train_df.drop(columns=[self.target_feature], inplace=False))
        x_test = scaler.transform(self.test_df.drop(columns=[self.target_feature], inplace=False))

        model = LogisticRegression(random_state=2002)
        lr_cv = GridSearchCV(model, param_grid, cv=5)
        lr_cv.fit(x_train, y_train)
        print("Tuned logistic Parameters: {}".format(
            lr_cv.best_params_))
        best_model = LogisticRegression(random_state=2002, **lr_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Logistic Regression")

    def knn(self):
        print("KNN")
        param_grid = [{'p': [1,2],
                       'weights': ['uniform', 'distance'],
                       'metric': ['euclidean', 'manhattan']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        errors = []
        for i in range(250):
            knn = KNeighborsClassifier(n_neighbors=i+1)
            knn.fit(x_train,y_train)
            y_pred = knn.predict(x_test)
            errors.append(accuracy_score(y_test,y_pred))

        plt.plot(np.arange(1,251), errors)
        plt.grid(True)
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Accuracy")
        plt.show()
        min_error = max(errors)
        print(errors)
        optimal_k = errors.index(min_error) + 1

        print(optimal_k)
        knn = KNeighborsClassifier(n_neighbors=optimal_k)

        knn_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_cv.fit(x_train, y_train)
        print("Tuned KNN Parameters: {}".format(
            knn_cv.best_params_))
        best_model = KNeighborsClassifier(n_neighbors=optimal_k, **knn_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "KNN")

    def svm(self):
        print("SVM")
        # linear kernel, polynomial kernel, radial base kernel
        param_grid = [{'kernel': ['linear', 'poly','rbf']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        svm = SVC(probability=True, random_state=2002)

        svm_cv = GridSearchCV(svm, param_grid, cv=5)
        svm_cv.fit(x_train, y_train)
        print("Tuned SVM Parameters: {}".format(
            svm_cv.best_params_))
        best_model = SVC(kernel='linear', probability=True)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Support Vector Machine")

    def naive_bayes(self):
        print("Naive Bayes")
        param_grid = [{'var_smoothing': [1e-9, 1e-7, 1e-5, 1e-3]}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        nb = GaussianNB()
        nb_cv = GridSearchCV(nb, param_grid, cv=5)
        nb_cv.fit(x_train, y_train)
        print("Tuned SVM Parameters: {}".format(
            nb_cv.best_params_))
        best_model = GaussianNB(**nb_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, f"Gaussian Naive Bayes")

    def create_keras_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(optimizer='adam',
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def neural_network(self):
        print("NN")
        y_train = self.train_df[self.target_feature].to_numpy()
        y_test = self.test_df[self.target_feature].to_numpy()
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False).to_numpy()
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False).to_numpy()

        best_model = self.create_keras_model()
        model_history = best_model.fit(x_train, y_train, epochs=30, validation_split=0.2)
        val_per_epoch = model_history.history['val_accuracy']
        optimal_epochs = val_per_epoch.index(max(val_per_epoch)) + 1
        print(f"Optimal number of epochs is {optimal_epochs}")

        optimal_model = self.create_keras_model()
        optimal_model.fit(x_train, y_train, epochs=optimal_epochs, validation_split=0.2)
        y_prob = best_model.predict(x_test)
        y_pred = (y_prob >= 0.5).astype("int")
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_prob, "Neural Network")

if __name__ == "__main__":
    obj = ClassificationAnalysis()

