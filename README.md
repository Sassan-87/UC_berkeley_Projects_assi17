# Bank Marketing Classification

This project compares several machine-learning classifiers for predicting whether a bank customer will subscribe to a term deposit after a marketing campaign.

## Notebook

Open the complete analysis here:

[View the Jupyter Notebook](prompt_III.ipynb)

## Dataset

The analysis uses the UCI Bank Marketing dataset, specifically `data/bank-additional-full.csv`. The data represents 17 telephone marketing campaigns conducted by a Portuguese bank between May 2008 and November 2010.

The target variable is `y`, indicating whether the customer subscribed to a term deposit:

- `no`: 88.73% of records
- `yes`: 11.27% of records

There are no true missing values, but several categorical columns contain `unknown`. The value `999` in `pdays` indicates that the customer had not previously been contacted.

## Objective

Build a classification model that can identify customers who are more likely to subscribe before a call is made. This could help the bank focus campaign resources on higher-probability customers.

## Methodology

- Encoded categorical variables using one-hot encoding.
- Excluded `duration` from the predictive features because it is only known after the call and would not be available for a pre-call prediction.
- Used accuracy for the initial comparison and ROC AUC for model selection because the target classes are imbalanced.
- Compared Logistic Regression, K-Nearest Neighbors, Decision Tree, and Support Vector Machine classifiers.
- Tuned selected hyperparameters with three-fold `GridSearchCV` using ROC AUC as the scoring metric.

## Findings

### Baseline and initial models

The majority-class baseline achieved `88.73%` test accuracy. Using only basic bank-client features (`age`, `job`, `marital`, `education`, `default`, `housing`, and `loan`) produced limited improvement:

| Model | Test accuracy |
| --- | ---: |
| Logistic Regression | 0.8873 |
| KNN | 0.8744 |
| Decision Tree | 0.8656 |
| SVM | 0.8873 |

These results demonstrate why accuracy alone is not sufficient for this problem: a model can appear strong while mostly predicting the majority class.

### Tuned models with expanded features

After adding campaign, contact, previous-outcome, and social/economic context features, standardizing Logistic Regression inputs, and tuning hyperparameters, performance improved:

| Model | Best parameters | Test accuracy | Test ROC AUC |
| --- | --- | ---: | ---: |
| Logistic Regression | `C=0.1` | 0.8972 | 0.7806 |
| Decision Tree | `max_depth=5` | 0.8962 | 0.7675 |
| SVM | `C=10`, `kernel='rbf'` | 0.8948 | 0.7441 |
| KNN | `n_neighbors=25` | 0.8941 | 0.7646 |

Logistic Regression was the strongest overall model in this experiment, with the highest test accuracy and ROC AUC. The results also show that adding relevant customer and campaign context was more valuable than using the basic client-information feature set alone. The notebook also includes exploratory plots, precision-recall diagnostics, a confusion matrix, and coefficient interpretation.

## Limitations

- The positive class is relatively small, so precision, recall, ROC AUC, and possibly precision-recall AUC should be considered alongside accuracy.
- The full-feature experiment uses a different train/test split from the initial comparison, so the two sets of results should be interpreted as directional rather than as a perfectly controlled benchmark.
- `unknown` categories and the special `pdays=999` value may require additional domain-specific treatment in a production model.
- The notebook is an educational comparison, not a deployed marketing system.

## Files

- `prompt_III.ipynb` - analysis, preprocessing, model training, tuning, and visualizations
- `data/bank-additional-full.csv` - full dataset used in the notebook
- `data/bank-additional.csv` - additional dataset supplied with the project
- `data/bank-additional-names.txt` - feature descriptions
- `CRISP-DM-BANK.pdf` - supporting research paper

## Findings and Recommendations

### Findings

1. The target is imbalanced: 88.73% of customers did not subscribe and 11.27% did subscribe.
2. The majority-class baseline achieved 88.73% accuracy, so accuracy alone is not an adequate success criterion.
3. Models using only basic bank-client information performed near the baseline.
4. Adding campaign, contact, previous-outcome, and economic-context features improved test accuracy to approximately 89.7%.
5. Tuned Logistic Regression was the strongest overall model in this experiment, with test accuracy of 0.8972 and test ROC AUC of 0.7806.
6. The age and job visualizations suggest that subscription rates vary across customer groups, but these patterns should be treated as predictive associations rather than causal conclusions.

### Recommendations and Next Steps

- Use the tuned Logistic Regression model to rank customers by predicted subscription probability rather than using a simple yes/no cutoff.
- Select an operating threshold using campaign capacity and the relative cost of missed subscribers versus unnecessary calls.
- Report precision, recall, F1 score, ROC AUC, and average precision alongside accuracy for future model comparisons.
- Validate the model with a consistent stratified holdout or time-based split before comparing models or deploying it.
- Investigate how `unknown` categories and the special `pdays=999` value should be handled with the bank's domain experts.
- Monitor performance over time because campaign response rates and economic conditions may change.