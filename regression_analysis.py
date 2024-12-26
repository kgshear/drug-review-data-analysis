import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

class RegressionAnalysis():
    train_filepath = "data/filtered/train_data_filtered_useful.csv"
    test_filepath = "data/filtered/test_data_filtered_useful.csv"
    def __init__(self):
        self.get_processed_data()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')
        elim_features = self.stepwise_regression()
        model = self.regression_analysis(elim_features)
        self.t_test_analysis()
        self.f_test_analysis(elim_features)
        self.confidence_interval_analysis(elim_features)

    def regression_analysis(self, eliminated_features):
        train_data = self.train_df.copy().sort_values(by=['usefulStandardized'])
        x_train = train_data.drop(columns=["usefulStandardized", *eliminated_features])
        y_train = train_data["usefulStandardized"]
        test_data = self.test_df.copy().sort_values(by=['usefulStandardized'])
        x_test = test_data.drop(columns=["usefulStandardized", *eliminated_features])
        y_test = test_data["usefulStandardized"]
        model = LinearRegression()
        model.fit(x_train, y_train)
        print(model.coef_)
        y_pred = model.predict(x_test)
        y_true = y_test
        subset_df = y_true[y_true > 1]

        print(subset_df)

        train_indices = np.arange(len(train_data))
        test_indices = np.arange(len(train_data), len(train_data) + len(test_data))
        plt.scatter(train_indices, y_train, label="Training Data")
        print(len(y_true), len(test_indices))
        plt.scatter(test_indices, y_pred, label="Test Predictions")
        plt.scatter(test_indices, y_true, label="Test Results")
        plt.title("Train, Test, and Predicted Values")
        plt.xlabel("Data Points")
        plt.ylabel("usefulStandardized")
        plt.legend()
        plt.grid(True)
        plt.show()
        return model

    def t_test_analysis(self):
        sig = 0.05
        t_table = PrettyTable()
        t_table.title = "T-test Analysis (alpha=0.05)"
        t_table.field_names = ["Feature", "t statistic", "p value", "Rejected?"]

        x_train = self.train_df.drop(columns=["usefulStandardized"])
        y_train = self.train_df["usefulStandardized"]
        model = sm.OLS(y_train, x_train).fit()
        p_values = model.pvalues
        t_stats = model.tvalues
        for val in t_stats.index:
            t_val = t_stats[val]
            p_val = p_values[val]
            if p_val < sig:
                isRejected="No"
            else:
                isRejected="Yes"
            t_table.add_row([f"{val}", f"{t_val:.3f}", f"{p_val:.3f}", f"{isRejected}"])
        print(t_table)

    def f_test_analysis(self, eliminated_features):
        x_train = self.train_df.drop(columns=["usefulStandardized", *eliminated_features])
        y_train = self.train_df["rating"]
        model = sm.OLS(y_train, x_train).fit()
        print("F-statistic:", model.fvalue)
        print("p-value for F-test:", model.f_pvalue)


    def confidence_interval_analysis(self, eliminated_features):
        x_train = self.train_df.drop(columns=["usefulStandardized", *eliminated_features])
        x_test = self.test_df.drop(columns=["usefulStandardized", *eliminated_features])
        y_train = self.train_df["usefulStandardized"]
        model = sm.OLS(y_train, x_train).fit()
        predictions = model.get_prediction(x_test)
        pred_mean = predictions.predicted_mean
        summary_frame = predictions.summary_frame(alpha=0.01)
        lower_bound = summary_frame["mean_ci_lower"]
        upper_bound = summary_frame["mean_ci_upper"]

        plt.fill_between(range(len(pred_mean)), lower_bound, upper_bound, color="blue", label="CI", alpha=.3)
        plt.scatter(range(len(pred_mean)), pred_mean, color="red", label="Predicted usefulStandardized", s=.3)

        plt.xlabel('# of Samples')
        plt.ylabel('usefulStandardized')
        plt.title('usefulStandardized Prediction with Confidence Interval')
        plt.legend()
        plt.grid(True)
        plt.show()

    def stepwise_regression(self):
        threshold = 0.05
        x_train = self.train_df.drop(columns=["usefulStandardized"], inplace=False)
        y_train = self.train_df["usefulStandardized"]
        x_train = sm.add_constant(x_train)
        mse_list = []
        aic_list = []
        bic_list = []
        r_adj_list = []
        r_two_list = []
        elim_pval_list = []
        elim_features = []
        total_features = x_train.drop(columns=["const"]).columns.tolist()
        features_so_far = []
        while True:
            model = sm.OLS(y_train, x_train).fit()
            p_values = model.pvalues.drop('const')
            max_p_value = p_values.max()
            aic_list.append(model.aic)
            bic_list.append(model.bic)
            r_adj_list.append(model.rsquared_adj)
            r_two_list.append(model.rsquared)
            mse_list.append(model.mse_model)
            elim_pval_list.append(p_values.max())
            features_so_far.append(p_values.idxmax())
            if max_p_value > threshold:
                feature_to_remove = p_values.idxmax()
                elim_features.append(feature_to_remove)
                x_train = x_train.drop(columns=[feature_to_remove])
                print(f"Removing {feature_to_remove} from model")
            else:
                for feature in total_features:
                    if feature not in features_so_far:
                        aic_list.append(model.aic)
                        bic_list.append(model.bic)
                        r_adj_list.append(model.rsquared_adj)
                        r_two_list.append(model.rsquared)
                        mse_list.append(model.mse_model)
                        elim_pval_list.append(p_values[feature])
                        features_so_far.append(feature)
                break
        print(f"Final summary is:\n {model.summary().as_text()}")
        reg_table = PrettyTable()
        reg_table.title = "Backwards Stepwise Regression Feature Elimination (alpha = 0.05)"
        reg_table.field_names = ["Feature", "AIC Value", "BIC Value", "MSE Value", "Adjusted R squared", "R Squared", "P-value", "Eliminated?"]
        count = len(mse_list)
        for num in range(count):
            isEliminated = "No"
            if features_so_far[num] in elim_features:
                isEliminated = "Yes"
            reg_table.add_row(
                [f"{features_so_far[num]}", f"{aic_list[num]:.3f}", f"{bic_list[num]:.3f}", f"{mse_list[num]:.3f}", f"{r_adj_list[num]:.3f}",
                 f"{r_two_list[num]:.3f}", f"{elim_pval_list[num]:.3f}", isEliminated])

        print(reg_table)
        return elim_features

if __name__ == "__main__":
    obj = RegressionAnalysis()
