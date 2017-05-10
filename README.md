## Tthe final project of Intro_to_Machine_Learning in Udacity 

## Dependencies:
numpy
scipy
scikit-learn
matplotlib

###1. Goal of this project – To identify who is the person of interest (POI) in Enron

###2. Exploration of the datsaset

In this dataset, there are 146 people and there are two major categories of features, namely financial feature and email features. Financial features are all numerical values, including salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock and director_fees. Similarly email features are also numerical values, including to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi and shared_receipt_with_poi. In sum, there are 19 features mentioned above regardless of the feature poi and the feature email_address which stores the email address of the person. 

As to the target feature “poi”, in this dataset, there are 18 people who are POI and 126 who aren’t. So, it is an unbalanced dataset. 

Carefully, there is an outlier in this dataset, that is “TOTAL”, whose features represent the total sum of the values of those feature. From the box plot of each feature, it can be evidently seen. It should be removed in advance. After removal, the boxplot would look like this one. (Refer to the report)

###3. Feature Selection 

In order to know which feature can be used to identify the POI, the box plots of each feature which show the comparison between the POI and non-POI can be helpful. 

From this plot, it can be speculated, the features, “salary”, “bonus”, “expenses”, “shared_receipt_with_poi”, “from_poi_to_this_person” and exercised_stock_options” may be helpful in classifying POI and nonPOI. The reason is that the spreading of the values of these features between two groups may be slightly more different than that of other features.

Another way to choose features is to use SelectKBest in sklearn, which will rank the importance of each feature according to the contribution to the accuracy. After running SelectKBest (input: all features), the scores are in the chart below. It can be suggested that “salary”, “bonus”, “expenses”, “restricted_stock”, “load_advances”, “deferred_income”, “total_payment”, “exercised_stock_options”, “long_term_incentive” and “shared_receipt_with_poi” (10 features, which scored above 5) are helpful in classification. Also, these results support the feature selection by box plot. (Refer to the report)


###4. Classification algorithm

After feature selection, 10 features are used to build up the model for identifying POI. Here, I tried Naïve Bayes, Decision Tree, Adaboost Decision Tree, and Logistic Regression. I got the two best f1 score, which are 0.32 and 0.33, by using Naïve Bayes and Adaboost respectively. (Refer to the report)

Also, I used PCA for all features, and get back 10 features from PCA for classification. With PCA, the two best f1 scores I got are from Naïve Bayes and Decision Tree. (Refer to the report)



