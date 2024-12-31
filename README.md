![image](https://github.com/user-attachments/assets/e4cba28d-25f8-40ea-b489-95c01f066dd2)
# Overview
<details> 
   <summary>Summary of Methods</summary>
  
  ### EDA/Feature Engineering
  Data Preprocessing, Feature Engineering, Sentiment Analysis, Natural Language Processing, Feature Reduction, Anomaly Detection, Feature Balancing
  ### Regression
  Linear Regression 
  ### Classification
  Hyperparameter Tuning, Decision Tree Classification (Pre-pruned and Post-pruned), Logistic Regression, K-Nearest Neighbors Classification, State Vector Machine Classification, Naive Bayes Classification, Neural Network Classification
  ### Clustering
  Unsupervised Learning, K-Means Clustering, DBSCAN CLusering, Association Rule Mining
</details>

### Code
This is a collection of preprocessing, anomaly reduction, regression, and classification methods performed on a dataset. The goal of this project was to find the best classifier for this dataset

### Dataset
The dataset used in this project is derived from reviews on the [Drugs.com](https://www.drugs.com/) website. This dataset provides reviews from patients on drugs that have been used to treat different conditions.
It includes ratings from patients and count of how many patients found that review useful. It contains 215,063 observations. The columns available are id, drug name, condition, review, rating, date, and usefulCount. For classification, a binary form of rating was used as the target variable. The actual dataset can be found [here](https://doi.org/10.24432/C5SK5S)
| Feature | Description | Type |
| --- | --- | --- |
| drugName | the name of the drug | Categorical |
| Condition | the ailment that the patient is taking the drug for | Categorical |
| Review | string describing the patient’s opinion on the drug | Categorical |
| Date | string that indicates when the review was posted | Categorical |
| Rating | integer between 1-10 that describes how much the patient likes the drug | Numerical |
| usefulCount | integer that represents the number of times the review has been upvoted by another user | Numerical |

# Exploratory Data Analysis and Feature Engineering
### Imputing 
There were originally 2,365 missing values in the dataset, all of which were conditions. These values were imputed by creating a mapping from drugs to their most common condition. After this, there were only 74 missing values. 
### Creating Features
Many features were created from the data. There is a feature named “word_count”, which contains a standardized version of the number of words in the review, as well as a feature named “upper_count”, which contains a standardized version of the number of uppercase words in the review. There is also a feature named “reviewFrequency”, which is a binary value based on whether the number of reviews for that drug is greater than the average number of reviews. Features that contained text, such as “condition” and “date”, were label encoded. “rating” was converted to a binary value based on whether the rating was low (1-5) or high (6-10). “usefulCount” was standardized.

Sentiment Analysis was used on the dataset to create a label denoting the connotation of the review (negative, neutral, or positive). To do this, a [BERT](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) large language model trained on twitter review was used.

### Dimensionality Reduction
HDBSCAN, a density-based clustering algorithm, was used to detect anomalies in the training dataset and remove noise. After using this technique, 5,418 rows were removed from the dataset. 

Random Forest Regression and Variance Inflation Factor (VIF) were utilized to detect multicolinearity in the dataset and remove it

### Balancing
The dataset was very unbalanced, with high ratings (1) having 115,543 instances and low ratings (0) having 51,031 instances. The minority class was oversampled using the  Synthetic Minority Oversampling Technique (SMOTE). This resulted in both classes having an equal amount of instances.
 
# Regression Analysis
Linear Regression was performed on the continuous variable “usefulStandardized”, which is a standardized version of “usefulCount”. This was done to determine which features have a statistically significant relationship with this feature.

Backward stepwise regression was performed on the dataset to remove features that do not improve the model. When running this algorithm on this dataset, no features were removed. 

An F-test and the R-squared values were calculated to determine how well the linear regression model fit the data. The R-squared value was extremely low, suggesting that the linear regression model was a bad fit for this data

# Classification Analysis

A variety of classifiers were applied to the dataset to predict the target variable, “rating”. This is a binary classification problem, since rating is either zero or one depending on whether the rating was high or low. Hyperparameter tuning was done to each classifier to find the best parameters. Stratified k-fold cross validation used during the hyperparameter tuning to verify that the models can perform well on unseen data splits.  Evaluation metrics, including precision, recall, specificity, and F-score were applied to each classifier to determine their performance. In addition, a confusion matrix and a receiver operating characteristic (ROC) curve was applied to each classifier to visualize this performance. 

The classifiers compared were Pre-pruned Decision Tree, Post-Pruned Decision Tree, Logistic Regression, K-Nearest Neighbors, State Vector Machine, Naive Bayes, and a Neural Network. By evaluating these classifiers through various metrics, it was found that Logistic Regression and the Neural Network have the best performance. 

# Future Work
In the future, there are a lot of ways I could improve the performance of this classification. I suspect that the dataset is too complex, which is why the clustering performed so poorly. I think that simplifying “condition” so that it will have less unique values would improve the results. I could possibly map it to another dataset with broader condition categories or a severity rating. I had difficulty finding a public dataset that provided this information. I also think that transforming the review into text embeddings and using those to train the model would improve results. I experimented with this using a word2vec algorithm but made a small number of embeddings due to hardware constraints. I found that this resulted in an increase of collinearity, so I removed these embeddings. I think that creating a set of 100-300 text embeddings would help with classifying whether the rating is high or not. I also think that it could be interesting to create a binary feature that represents whether the drug is over the counter or prescription. I also attempted to find a dataset with this information that I could map this one to, but I struggled to find all of that information in one place.

