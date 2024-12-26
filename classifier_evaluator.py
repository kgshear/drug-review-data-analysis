from prettytable import PrettyTable
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, roc_auc_score, ConfusionMatrixDisplay, \
    confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


class ClassifierEvaluator:

    def __init__(self):
        pass

    def evaluate(self, y_pred, y_true, y_prob, type):
        self.display_confusion_matrix(y_pred, y_true, type)
        self.display_result(y_pred, y_true, type)
        self.display_roc_and_auc(y_true, y_prob, type)

    def display_confusion_matrix(self, y_pred, y_true, type):
        confus_matrix = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=confus_matrix, display_labels=[0, 1])
        display.plot()
        plt.title(f"Confusion Matrix for {type} Classifier")
        plt.show()

    def display_result(self, y_pred, y_true, type):
        result_table = PrettyTable()
        result_table.title = f"{type} Classifier Results"
        result_table.field_names = ["Type", "Precision", "Recall", "Specificity", "F-score", "Accuracy"]
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        specificity = recall_score(y_true, y_pred, pos_label=0)
        f1 = f1_score(y_true,y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        result_table.add_row([f"{type}", f"{precision}", f"{recall}", f"{specificity}", f"{f1}", f"{accuracy}"])
        print(result_table)


    def display_roc_and_auc(self, y_true, y_prob, type):
        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.grid(True)
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {type} Classifier")
        plt.show()