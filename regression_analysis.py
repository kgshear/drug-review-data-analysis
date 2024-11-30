import pandas as pd
from prettytable import PrettyTable
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind
from scipy.stats import linregress, f_oneway
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
 # lecture 8
class RegressionAnalysis():
    train_filepath = "data/processed/train_data_processed.csv"
    test_filepath = "data/processed/test_data_processed.csv"
    def __init__(self):
        self.get_processed_data()

    def get_processed_data(self):
        print("getting data")
        self.test_df = pd.read_csv(self.test_filepath, sep=',')
        self.train_df = pd.read_csv(self.train_filepath, sep=',')

    def regression_analysis(self, eliminated_features):
        x_train = self.train_df.drop(columns=["rating", *eliminated_features])
        y_train = self.train_df["rating"]
        x_test = self.test_df.drop(columns=["rating", *eliminated_features])
        y_test = self.test_df["rating"]
        model = LinearRegression()
        model.fit(x_train, y_train)
        print(model.coef_)
        y_pred = model.predict(x_test)
        y_true = y_test
        plt.plot(range(x_train), y_train, label="Training Data")
        plt.plot(range(y_pred), y_pred, label="Test Predictions")
        plt.plot(range(y_pred), y_true, label="Test Results")
        plt.title("Train, Test, and Predicted Values")
        plt.xlabel("Data Points")
        plt.ylabel("Rating")
        plt.legend()
        plt.grid(True)
        plt.show()


    def t_test_analysis(self):
        sig = 0.05
        t_table = PrettyTable()
        t_table.title = "T-test Analysis (alpha=0.05)"
        t_table.field_names = ["Feature", "t statistic", "p value", "Rejected?"]
        features = self.test_df.drop(columns=["rating"], inplace=False).columns
        for feature in features:
            feature_mean = np.mean(self.test_df[feature])
            test_result = linregress(self.test_df[feature], self.test_df["rating"])
            t_val = test_result.slope / (test_result.sterr/ np.sqrt(np.sum((self.test_df[feature] - feature_mean)**2)))
            p_val = test_result.pvalue
            isRejected = "No"
            if p_val > sig:
                isRejected="No"
            else:
                isRejected="Yes"
            t_table.add_row([f"{feature}", f"{t_val:.5f}", f"{p_val:.3f}", f"{isRejected}"])
        print(t_table)


            # feature_mean = np.mean(self.test_df[feature])
            # target_mean = np.mean(self.test_df["rating"])
            # beta_hat = np.cov(self.test_df[feature], self.test_df["rating"])[0,1] / (np.std(self.test_df[feature], ddof=1)**2)
            # alpha = target_mean - beta_hat * feature_mean
            # y_hat = alpha + self.test_df[feature] * beta_hat
            # sse = np.sum((self.test_df["rating"]-y_hat)**2)
            # se = np.sqrt(sse / (len(self.test_df)-2))
            # t_stat = beta_hat / (se/(np.sqrt((self.test_df[feature] - feature_mean))))
        #can only eval one regression coefficient at a time
        # bhat = slope
        # SE(bhat) = rse/sum(x-xmean)^2
        # t stat = (bhat - 0) / se(bhat)
        # print(np.var(y_pred), np.var(y_true))
        # alpha = 0.05
        # #this assumes variances are equal
        # t_stat, p_val, df = ttest_ind(a=y_pred, b=y_true)
        # if p_val > alpha:
        #     print(f"Fail to reject null hypothesis: p value of {p_val:.4f}, alpha of {alpha}, t statistic of {t_stat:.4f}")
        # else:
        #     print(f"Reject null hypothesis: p value of {p_val:.4f}, alpha of {alpha}, t statistic of {t_stat:.4f}")
        #pretty table that shwos whether it was rejected for each val

    def f_test_analysis(self):
        sig = 0.05
        x_train = sm.add_constant(self.train_df.drop(columns=["rating"], inplace=False))
        y_train = self.train_df["rating"]
        model = sm.OLS(y_train, x_train).fit()
        summary = model.summary()



    def confidence_interval_analysis(self, model):
        predictions = model.get_prediction(self.test_df.drop(columns=["rating"], inplace=False))
        pred_mean = predictions.predicted_mean
        conf_int = predictions.conf_int(alpha=0.05)
        lower_bound = conf_int[:, 0]
        upper_bound = conf_int[:, 1]
        plt.plot(range(len(pred_mean)), pred_mean, color="red", label='Predicted Rating')
        plt.fill_between(range(len(pred_mean)), lower_bound, upper_bound, label='CI')
        plt.xlabel('# of Samples')
        plt.ylabel('Rating')
        plt.title('Rating Prediction with Confidence Interval')
        plt.legend()
        plt.grid(True)
        plt.show()

    def stepwise_regression(self):
        threshold = 0.01
        x_train = self.train_df.drop(columns=["rating"], inplace=False)
        y_train = self.train_df["rating"]
        x_train = sm.add_constant(x_train)
        mse_list = []
        aic_list = []
        bic_list = []
        r_adj_list = []
        r_two_list = []
        elim_pval_list = []
        elim_features = []
        cur_features = x_train.columns.tolist()
        total_features = x_train.columns.tolist()
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
            if max_p_value > threshold:
                feature_to_remove = p_values.idxmax()
                elim_features.append(feature_to_remove)
                x_train = x_train.drop(columns=[feature_to_remove])
                print(f"Removing {feature_to_remove} from model")
            else:
                break
        print(f"Final summary is:\n {model.summary().as_text()}")
        reg_table = PrettyTable()
        reg_table.title = "Backwards Stepwise Regression Feature Elimination (alpa = 0.01)"
        reg_table.field_names = ["AIC Value", "BIC Value", "MSE Value", "Adjusted R squared", "R Squared", "P-value", "Feature", "Eliminated?"]
        count = len(elim_features)
        for num in range(count):
            isEliminated = "No"
            if total_features[num] in elim_features:
                isEliminated = "Yes"
            reg_table.add_row(
                [f"{aic_list[num]:.3f}", f"{bic_list[num]:.3f}", f"{mse_list[num]:.3f}", f"{r_adj_list[num]:.3f}",
                 f"{r_two_list[num]:.3f}", f"{elim_pval_list[num]:.3f}", f"{total_features[num]}", isEliminated])
        return elim_features