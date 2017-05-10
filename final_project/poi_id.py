#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus', 'expenses','loan_advances','long_term_incentive','exercised_stock_options',
'shared_receipt_with_poi','from_poi_to_this_person','from_this_person_to_poi'] # You will need to use more features

features_list = ['poi','salary','bonus', 'expenses','director_fees','to_messages','restricted_stock'
,'loan_advances','deferred_income','deferral_payments','long_term_incentive'
,'from_messages','total_payments','restricted_stock_deferred','other','exercised_stock_options','shared_receipt_with_poi',
'from_poi_to_this_person','from_this_person_to_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Backgroun check (Data Explore Analysis, DEA)
# How many people in the dataset
all_people = [name for name in data_dict.keys()]
#print("Number of people:", len(all_people)) # 146 people in this dataset
#print("\n".join([name for name in all_people])) 

# How many POI in the dataset
all_POI_names = [name for name in data_dict.keys() if data_dict[name]['poi'] == 1]
all_non_POI_names = [name for name in data_dict.keys() if data_dict[name]['poi'] == 0]
print("Number of POI:", len(all_POI_names))
print("Number of non-POI:", len(all_non_POI_names))

# How many features in the dataset
all_features = [feature for feature in data_dict[all_people[0]].keys()]
print("Number of features:", len(all_features)) # 21 features (including POI, email_address)
#print("\n".join([feature for feature in all_features])) 

# remove non-numerical features for training (poi, email_address)

train_features = [feature for feature in data_dict[all_people[0]].keys() if feature not in ["poi","email_address"]]
print("Number of Features for Training:", len(train_features)) # It should be 19

# Look into the descriptive statitics of all train features
def printDescriptives(features, data_dict):
	'''
		calculate descriptive statistics for train features, then print & return
	'''
	all_stats = []
	for feature in features:
		feature_data = featureFormat(data_dict, [feature], sort_keys = True)
		mean_data = np.mean(feature_data)
		std_data = np.std(feature_data)
		confidience_interval = (mean_data - 2*std_data, mean_data + 2*std_data)
		median_data = np.median(feature_data)
		per_25, per_75 = np.percentile(feature_data, [25., 75.])
		min_value = np.min(feature_data)
		max_value = np.max(feature_data)
		all_stats.append({"feature": feature,"mean": mean_data, "std": std_data, "CI": confidience_interval,
			"min": min_value,"max": max_value, 
			"median": median_data, "percentile_75": per_75, "percentile_25": per_25})
		
		#print out
		print("feature:{:s} -mean:{:.2f}, std:{:.2f}, CI:{:.2f}-{:.2f}, min_max_range:{:.2f}-{:.2f},median:{:.2f}, per75:{:.2f}, per25:{:.2f}".format(feature, mean_data, std_data, confidience_interval[0], confidience_interval[1],min_value, max_value, median_data, per_75, per_25))
	return all_stats
# Show all the stats of train features
#all_stats = printDescriptives(train_features, data_dict)

#Draw boxplot for all numerical features
'''
import matplotlib.pyplot as plt
for i, feature in enumerate(train_features):
	feature_data = featureFormat(data_dict, [feature])
	sp = plt.subplot(4,5,(i+1))
	box = plt.boxplot(feature_data, meanline = True)
	plt.title(feature, fontsize = 5)
	sp.axes.get_yaxis().set_visible(False)
	sp.axes.get_xaxis().set_visible(False)
plt.show()
'''


### Task 2: Remove outliers 

# Remove the most evident outlier 
# find the one that got the highest salary
#outlier_name = filter(lambda x: data_dict[x]["salary"] == all.stats[0]["max"], all_people)
#print("Outlier:", outlier_name[0]) # 'TOTAL'
data_dict.pop('TOTAL', None)

# Data Explore Again
#all_stats = printDescriptives(train_features, data_dict)

'''
# boxplot again
import matplotlib.pyplot as plt
for i, feature in enumerate(train_features):
	feature_data = featureFormat(data_dict, [feature])
	sp = plt.subplot(4,5,(i+1))
	box = plt.boxplot(feature_data, meanline = True)
	plt.title(feature, fontsize = 5)
	sp.axes.get_yaxis().set_visible(False)
	sp.axes.get_xaxis().set_visible(False)
plt.show()
'''

# Back to Task 1: Select Features

def getFeaturesWithLabel(feature_name, data_dict, label):

	negative_feature = []
	positive_feature = []
	for name in data_dict.keys():
		if data_dict[name][label] == 0:
			value = data_dict[name][feature_name]
			if value == 'NaN':
				value = 0
			negative_feature.append(float(value))
		elif data_dict[name][label] == 1:
			value = data_dict[name][feature_name]
			if value == 'NaN':
				value = 0
			positive_feature.append(float(value))
	return np.array(positive_feature), np.array(negative_feature)

def boxplotForSelectingFeatures(features_list, data_dict, label, subplot_attributes):
	'''
		boxplot for feature selection (to compare features between positve and negative examples)
	'''
	num_subplots = subplot_attributes[0]
	num_x_axis = subplot_attributes[1]
	num_y_axis = subplot_attributes[2]

	for i, feature_name in enumerate(features_list):
		pos_feature, neg_feature = getFeaturesWithLabel(feature_name, data_dict, label)
		sp = plt.subplot(num_x_axis,num_y_axis,(i+1))
		box = plt.boxplot([neg_feature, pos_feature], meanline = True)
		print(neg_feature)
		print(pos_feature)
		plt.title(feature_name, fontsize = 5)
		sp.axes.get_yaxis().set_visible(False)
		sp.axes.get_xaxis().set_visible(False)
	plt.show()

#boxplotForSelectingFeatures(train_features, data_dict, 'poi', (20, 4, 5))



### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# Feature Selection with SelectKBest
'''
from sklearn.feature_selection import SelectKBest, f_classif

num_k = 10
selector = SelectKBest(f_classif, num_k)
selector.fit(features, labels)
features = selector.transform(features)
'''
# Feature Selection with PCA
from sklearn.decomposition import RandomizedPCA

pca = RandomizedPCA(n_components = 10, whiten = True).fit(features)

features = pca.transform(features)
print(features.shape)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble	import AdaBoostClassifier


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import GridSearchCV
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

#svm_parameters = {'C':[1, 2, 5, 10], 'gamma': [0.001, 0.005, 0.01, 0.05, 0.1]}
#tree_parameters = {'min_samples_split':[2, 3, 5, 7, 9, 10], 'min_samples_leaf':[1, 2, 3, 5, 7, 9, 10]}
#knn_parameters = {'n_neighbors': [3, 4, 5, 6, 7]}
ada_parameters = {'n_estimators':[25, 30, 35, 40, 45, 50, 55, 60], 
'learning_rate':[0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1.0]}
#svm = SVC()
#tree = DecisionTreeClassifier(random_state = 42)
#knn = KNeighborsClassifier(n_neighbors = 3)
clf = AdaBoostClassifier(n_estimators = 25, learning_rate = 0.04, random_state = 42)
#clf = GridSearchCV(ada, ada_parameters)

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(labels_test, pred)
prec = precision_score(labels_test, pred)
rec = recall_score(labels_test, pred)
#important_features = clf.feature_importances_
#most_important_feature = np.amax(important_features, axis = 0)
#most_important_feature_loc = np.argmax(important_features, axis = 0)
#print(most_important_feature, most_important_feature_loc)
#print(clf.best_params_)

print("Accuracy:", acc, "; precision:", prec, "; recall:",rec)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)