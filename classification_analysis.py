from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from classifier_evaluator import ClassifierEvaluator


class ClassificationAnalysis:
    train_filepath = "data/filtered/train_data_filtered.csv"
    test_filepath = "data/filtered/test_data_filtered.csv"
    target_feature = "condition"
    test_df = None
    train_df = None
    def __init__(self, df):
        self.get_processed_data()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')

    # def kfold_cross_val(self, model):
    #     data = pd.concat([self.test_df, self.train_df], ignore_index=True)
    #     y = data[self.target_feature]
    #     x = data.drop(columns=[self.target_feature])
    #     skf = StratifiedKFold(n_splits=5)
    #     best_indexes = []
    #     best_accuracy = 0
    #     for train_index, test_index in skf.split(x,y):
    #         X_train, X_test = x[train_index], x[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         new_model = model(random_state=2002)
    #         new_model.fit(X_train, y_train)
    #         y_pred = model.predict(X_test)
    #         accuracy = accuracy_score(y_test, y_pred)
    #         if accuracy > best_accuracy:
    #             best_accuracy = accuracy
    #             best_indexes = [train_index, test_index]
    #     return best_indexes[0], best_indexes[1]

    def decision_tree(self):
        param_grid = [{'max_depth': [1, 2, 3, 4, 5],
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

        path = model.cost_complexity_pruning_path(x_train, y_train)
        alphas = path.ccp_alphas
        train_scores = []
        test_scores = []

        for alpha in alphas:
            new_model = DecisionTreeClassifier(random_state=2002, ccp_alpha=alpha)
            new_model.fit(x_train, y_train)
            train_scores.append(accuracy_score(y_train, new_model.predict(x_train)))
            test_scores.append(accuracy_score(y_test, new_model.predict(x_test)))

        best_alpha_idx = np.argmax(test_scores)
        best_alpha = alphas[best_alpha_idx]
        print(f"Optimal alpha: {best_alpha}")

        best_model = DecisionTreeClassifier(random_state=5805, **dt_cv.best_params_, ccp_alpha=best_alpha)
        best_model.fit(x_train, y_train)
        train_accuracy = accuracy_score(y_train, best_model.predict(x_train))
        test_accuracy = accuracy_score(y_test, best_model.predict(x_test))

        print(f"The train accuracy is {train_accuracy:.2f}")
        print(f"The test accuracy is {test_accuracy:.2f}")
        plt.figure(figsize=(16, 16))
        plot_tree(best_model, feature_names=x_train.columns, filled=True,
                  fontsize=14)
        plt.show()
        y_pred = best_model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred,y_test,y_pred_prob,"Decision Tree")

    def logistic_regression(self):
        param_grid = [{'max_iter': [80, 90, 100, 110, 120],
                       'C': [0.001,0.01,0.1,1,10,100],
                       'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                       'penalty': [None, 'l2', 'l1', 'elasticnet']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        model = LogisticRegression(random_state=2002)
        lr_cv = GridSearchCV(model, param_grid, cv=5)
        lr_cv.fit(x_train, y_train)
        print("Tuned DT Parameters: {}".format(
            lr_cv.best_params_))
        best_model = DecisionTreeClassifier(random_state=2002, **lr_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Logistic Regression")

    def knn(self):
        param_grid = [{'p': [1,2],
                       'weights': ['uniform', 'distance'],
                       'metric': ['euclidean', 'manhattan']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        errors = []
        for i in range(30):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(x_train,y_train)
            y_pred = knn.predict(x_test)
            errors.append(np.mean(y_pred != y_test))
        min_error = min(errors)
        optimal_k = errors.index(min_error)


        knn = KNeighborsClassifier(n_neighbors=optimal_k)

        knn_cv = GridSearchCV(knn, param_grid, cv=5)
        knn_cv.fit(x_train, y_train)

        best_model = KNeighborsClassifier(n_neighbors=optimal_k, **knn_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "KNN")

    def svm(self):
        # linear kernel, polynomial kernel, radial base kernel
        param_grid = [{'kernel': ['linear', 'poly','rbf']}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        svm = SVC(random_state=2002)

        svm_cv = GridSearchCV(svm, param_grid, cv=5)
        svm_cv.fit(x_train, y_train)

        best_model = SVC(**svm_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Support Vector Machine")

    def naive_bayes(self):
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        model_types = [GaussianNB, MultinomialNB, BernoulliNB]
        text_list = ["Gaussian", "Multinomial", "Bernoulli"]
        errors = []
        for model in model_types:
            new_model = model()
            new_model.fit(x_train,y_train)
            y_pred = new_model.predict(x_test)
            errors.append(np.mean(y_pred != y_test))
        min_error = min(errors)
        best_model = model_types[errors.index(min_error)]()
        best_model.fit(x_train,y_train)
        y_pred = best_model.predict(y_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, f"{text_list[errors.index(min_error)]} Naive Bayes")

    def random_forest_bagging(self):
        #bagging, stacking, boosting
        param_grid = [{'n_estimators':  [80, 90, 100, 110, 120],
                       'criterion': ['gini', 'entropy', 'log_loss'],
                       'max_depth': [1, 2, 3, 4, 5],
                       'min_samples_split': [2, 20, 30, 40],
                       'min_samples_leaf': [1, 10, 20, 30],
                       'max_features': ['sqrt', 'log2', None]}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        rf = RandomForestClassifier(random_state=2002)

        rf_cv = GridSearchCV(rf, param_grid, cv=5)
        rf_cv.fit(x_train, y_train)

        best_model = RandomForestClassifier(**rf_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Random Forest Bagging")

    def random_forest_stacking(self):
        #bagging, stacking, boosting
        param_grid = [{'rf__n_estimators':  [80, 90, 100, 110, 120],
                       'rf__criterion': ['gini', 'entropy', 'log_loss'],
                       'rf__max_depth': [1, 2, 3, 4, 5],
                       'rf__min_samples_split': [2, 20, 30, 40],
                       'rf__min_samples_leaf': [1, 10, 20, 30],
                       'rf__max_features': ['sqrt', 'log2', None],
                       'final_estimator__max_iter': [80, 90, 100, 110, 120],
                       'final_estimator__C': [0.001, 0.01, 0.1, 1, 10, 100],
                       'final_estimator__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                       'final_estimator__penalty': [None, 'l2', 'l1', 'elasticnet']
                       }]

        estimator_params = [('rf', RandomForestClassifier()), ('nb', MultinomialNB())]
        meta_model = LogisticRegression()
        clf = StackingClassifier(estimators=estimator_params, final_estimator=meta_model)

        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)

        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
        grid_search.fit(x_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Random Forest Stacking")

    def random_forest_boosting(self):
        #bagging, stacking, boosting
        param_grid = [{'n_estimators':  [80, 90, 100, 110, 120],
                       'criterion': ['gini', 'entropy', 'log_loss'],
                       'max_depth': [1, 2, 3, 4, 5],
                       'min_samples_split': [2, 20, 30, 40],
                       'min_samples_leaf': [1, 10, 20, 30],
                       'max_features': ['sqrt', 'log2', None]}]
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        rf = RandomForestClassifier(random_state=2002)

        rf_cv = GridSearchCV(rf, param_grid, cv=5)
        rf_cv.fit(x_train, y_train)

        base_estimator = RandomForestClassifier(**rf_cv.best_params_)

        boost_param_grid = [{'n_estimators':  [80, 90, 100, 110, 120],
                       'loss': ['log_loss', 'exponential'],
                       'learning_rate': [.01, .05, .1, .5, 1],
                       'criterion': ['friedman_mse', 'squared_error'],
                       'min_samples_leaf': [1, 10, 20, 30],
                       'min_samples_split': [2, 4, 6]}]
        boost_rf = AdaBoostClassifier(estimator=base_estimator)
        boost_cv = GridSearchCV(boost_rf, boost_param_grid, cv=5)
        boost_cv.fit(x_train, y_train)
        best_model = AdaBoostClassifier(estimator=base_estimator, **boost_cv.best_params_)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        y_pred_prob = best_model.predict_proba(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Random Forest Boosting")

    def neural_network(self):
        y_train = self.train_df[self.target_feature]
        y_test = self.test_df[self.target_feature]
        x_train = self.train_df.drop(columns=[self.target_feature], inplace=False)
        x_test = self.test_df.drop(columns=[self.target_feature], inplace=False)
        model = keras.Sequential()
        model.add(layers.Dense(256, "relu"))
        model.add(layers.Dense(16, "relu"))
        model.add(layers.Dense(8, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2)
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_pred_prob = model.predict(x_test)[:, 1]
        obj = ClassifierEvaluator()
        obj.evaluate(y_pred, y_test, y_pred_prob, "Random Forest Boosting")
        y_pred = model.predict(x_test)


        # Multi-layered perceptron
        pass