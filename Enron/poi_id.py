#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
data_dict.pop("TOTAL",0)
for i in data_dict:
    person = data_dict[i]
    if (all([person['from_poi_to_this_person'] != 'NaN',person['from_this_person_to_poi'] != 'NaN',person['to_messages'] != 'NaN',person['from_messages'] != 'NaN'])):
        fraction_from_poi = float(person["from_poi_to_this_person"]) / float(person["to_messages"])
        person["fraction_from_poi"] = fraction_from_poi
        fraction_to_poi = float(person["from_this_person_to_poi"]) / float(person["from_messages"])
        person["fraction_to_poi"] = fraction_to_poi
    else:
        person["fraction_from_poi"] = person["fraction_to_poi"] = 0

my_features_list = ['poi','salary','bonus','long_term_incentive','deferred_income','deferral_payments','total_payments','exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value','fraction_from_poi','fraction_to_poi']
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#feature selection
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

pca = PCA(n_components=2)
selector = SelectKBest(k=7)
combined_features= FeatureUnion([("pca", pca), ("univ_select", selector)])
combined_features.fit(features,labels).transform(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
pipe_nb=Pipeline([("features", combined_features),('clf',GaussianNB())])
test_classifier(pipe_nb, my_dataset, my_features_list)

from sklearn.linear_model import LogisticRegression
pipe_lgr=Pipeline([('sc',StandardScaler()),("features", combined_features),('clf',LogisticRegression())])
test_classifier(pipe_lgr, my_dataset, my_features_list)

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
tree = DecisionTreeClassifier()
pipe_tree=Pipeline([("features", combined_features),('clf',tree)])
test_classifier(pipe_tree, my_dataset, my_features_list)

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier()
pipe_rfc=Pipeline([("features", combined_features),('clf',forest)])
test_classifier(pipe_rfc, my_dataset, my_features_list)

from sklearn.ensemble import VotingClassifier
clf1=GaussianNB()
clf2=tree
clf3=forest
vc = VotingClassifier(estimators=[('gnb',clf1),('tr',clf2),('fr',clf3)])
pipe_vc=Pipeline([("features", combined_features),('clf',vc)])
test_classifier(pipe_vc, my_dataset, my_features_list)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)

#tuning pipe_nb
param_grid=[{"features__univ_select__k":range(1,10)}]
gs=GridSearchCV(estimator=pipe_nb,param_grid=param_grid,scoring="f1",cv=10)
gs.fit(features,labels)
combined_features= FeatureUnion([("pca", PCA(n_components=2)), ("univ_select", SelectKBest(k=9))])
combined_features.fit(features,labels).transform(features)
pipe_nb_best=Pipeline([("features", combined_features),('clf',GaussianNB())])
test_classifier(pipe_nb_best, my_dataset, my_features_list)

#tuning pipe_tree
# pipe_tree.get_params().keys()

param_grid=[{"features__univ_select__k":range(1,10),"clf__max_depth":range(1,10),"clf__min_samples_split":range(2,20,1)}]
gs=GridSearchCV(estimator=pipe_tree,param_grid=param_grid,scoring="f1",cv=10)
gs.fit(features,labels)

combined_features= FeatureUnion([("pca", PCA(n_components=2)), ("univ_select", SelectKBest(k=1))])
combined_features.fit(features,labels).transform(features)
tree_best = DecisionTreeClassifier(max_depth=4,min_samples_split=9)
pipe_tree_best=Pipeline([("features", combined_features),('clf',tree_best)])
test_classifier(pipe_tree_best, my_dataset, my_features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)