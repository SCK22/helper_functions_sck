"""Helper functions for Machine Learning"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, SGDClassifier, PassiveAggressiveClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score

class HelperFunctionsML:
	"""Helper functions for Machine Learning"""

def __init__(self, dataset):
	"""Helper Functions to do ML"""

	self.dataset = dataset

	def cat_num_extract(self):
		dataset = self.dataset
		"""This function returns the names of the Categorical and Nmeric attributes in the same order."""
		cat_cols = [i for i in dataset.columns.values if dataset[i].dtype in ['O', 'object']]
		num_cols = [i for i in dataset.columns.values if dataset[i].dtype not in ['O', 'object']]
		return(cat_cols, num_cols)

	def list_of_na_cols(self, dataset, per=0.3):
		"""This function will return the columns with na values"""

		self.dataset = dataset
		self.per = per

		na_cols = dataset.isnull().sum()[dataset.isnull().sum() > per]
		return(list(na_cols.index))

	def create_dummy_data_frame(self, dataset, categorical_attributes):
		"""This function returns a dataframe of dummified colums Pass the dataset and the column names."""
		return (pd.get_dummies(dataset[categorical_attributes]))

	def impute_numeric_col(self, column, method='median'):
		"""This function replaces the na values with the colum mean or median ,  based on the selection.
		Available values for method are 'meidan',  'mean'. default is median
		This function might not be needed anymore as sklearn .22 has KNN imputation which I would prefer to use."""
		
		if method == 'mean':
			return (column.fillna(axis=0, value=np.mean(column)))
		if method == 'median':
			return (column.fillna(axis=0, value=np.median(column)))

	def apply_model_predict_validate(self, model_name, X_train, y_train, X_validation, y_validation, test_data, feature_names):
		"""This function
		1.Applies the specified model to the train data.
		2.Validates on the validation set.
		3.Predicts on the test dataset
		4.Returns the predictions along with some scores.
		This is for classification."""

		# apply the model
		model_name, acc_val, prec_val, rec_val = np.array(['NA'])*4
		model_name.fit(X_train[feature_names], y_train)
	 	# evaluation metrics
		pred_val = model_name.predict(X_validation[feature_names])  # predict on the validation set
		acc_val = accuracy_score(y_pred=pred_val, y_true=y_validation)  # Accuracy for the validation set
		try:
			prec_val = precision_score(y_pred=pred_val, y_true=y_validation, average='macro')  # precision for the validation set
		except: pass
		try:
			rec_val = recall_score(y_pred=pred_val, y_true=y_validation, average='macro')  # recall for the validation set
		except : pass
		# test
		test_pred = model_name.predict(test_data[feature_names])  # predict on test data
		return(test_pred, "Model \'{}\'\n Accuracy:{}\n Precision:{}\n Recall:{}\n".format(model_name, acc_val, prec_val, rec_val))

	# Replace values in an attribute with other values
	def replace_attribute_values(self, dataset, target_attribute, originals, replace_with):
		""" This function takes a pandas series object and replaces the specified values with specified values."""
		if len(originals) == len(replace_with):
			for i in range(len(originals)):
				dataset[target_attribute].replace(originals[i], replace_with[i], inplace=True)
		elif len(originals != len(replace_with)):
			raise ValueError("replacement values do not match the size of originals")
		return dataset

	# def import_models(self):
	# 	"""Importing all models from sklearn classification."""
		
	# 	knn_model_3 = KNeighborsClassifier(3), 
	# 	knn_model_5 = KNeighborsClassifier(5), 
	# 	knn_model_7 = KNeighborsClassifier(7), 
	# 	knn_model_9 = KNeighborsClassifier(9), 
	# 	knn_model_11 = KNeighborsClassifier(11), 
	# 	knn_model_13 = KNeighborsClassifier(15), 

	# 	svm_linear=    SVC(kernel="linear",  C=0.025), 
	# 	svm_rbf = SVC(gamma=2,  C=1), 
	# 	gaussianprocess_model = GaussianProcessClassifier(1.0 * RBF(1.0)), 
	# 	decisiontree_model = DecisionTreeClassifier(max_depth=5), 
	# 	rf_model_10 = RandomForestClassifier(max_depth=5,  n_estimators=10,  max_features=1, verbose=True), 
	# 	rf_model_100 = RandomForestClassifier(max_depth=5,  n_estimators=100,  max_features=1, verbose=True), 
	# 	rf_model_250 = RandomForestClassifier(max_depth=5,  n_estimators=250,  max_features=1, verbose=True), 
	# 	rf_model_500 = RandomForestClassifier(max_depth=5,  n_estimators=500,  max_features=1, verbose=True), 
	# 	rf_model_750 = RandomForestClassifier(max_depth=5,  n_estimators=750,  max_features=1, verbose=True), 
	# 	rf_model_1000 = RandomForestClassifier(max_depth=5,  n_estimators=1000,  max_features=1, verbose=True), 
	# 	mlp_model = MLPClassifier(alpha=1, verbose=True), 
	# 	mlp_model_adaptive_learning = MLPClassifier(alpha=1, verbose=True, learning_rate = "adaptive", shuffle=True, validation_fraction= 0.3, hidden_layer_sizes = (100, 50, 100, 50)), 
	# 	adaboost_model_10 = AdaBoostClassifier(n_estimators = 10), 
	# 	adaboost_model_50 = AdaBoostClassifier(n_estimators = 50), 
	# 	adaboost_model_100 = AdaBoostClassifier(n_estimators = 100), 
	# 	adaboost_model_150 = AdaBoostClassifier(n_estimators = 150), 
	# 	adaboost_model_200 = AdaBoostClassifier(n_estimators = 200), 
	# 	adaboost_model_250 = AdaBoostClassifier(n_estimators = 250), 
	# 	adaboost_model_500 = AdaBoostClassifier(n_estimators = 500), 
	# 	adaboost_model_750 = AdaBoostClassifier(n_estimators = 750), 
	# 	adaboost_model_1000 = AdaBoostClassifier(n_estimators = 1000), 
	# 	adaboost_model_2000= AdaBoostClassifier(n_estimators = 2000), 
	# 	adaboost_model_3000= AdaBoostClassifier(n_estimators = 3000), 
	# 	adaboost_model_4000= AdaBoostClassifier(n_estimators = 4000), 
	# 	naive_bayes_model = GaussianNB(), 
	# 	quadrant_discriminant_analysis_model = QuadraticDiscriminantAnalysis(), 
	# 	linear_discriminant_analysis_model = LinearDiscriminantAnalysis(), 
	# 	# Passive_Aggressive_Classifier= PassiveAggressiveClassifier(C=1.0,  fit_intercept=True,  max_iter=None,  tol=None,  shuffle=True,  verbose=0,  n_jobs=1, 
	# 	#  random_state=None,  warm_start=False,  class_weight = None,  average=False,  n_iter=None), 
	# 	SGD_Classifier =SGDClassifier(), 
	# 	Radius_Neighbours_classifier = RadiusNeighborsClassifier(), 
	# 	Ridge_classifier_cv = RidgeClassifierCV(), 
	# 	Etree_Classifier=ExtraTreesClassifier(), 
	# 	Nu_svc = NuSVC(), 
	# 	models_dict = {'knn_model_3':knn_model_3, 
	# 					"knn_model_5":knn_model_5, 
	# 					"knn_model_7":knn_model_7, 
	# 					"knn_model_9":knn_model_9, 
	# 					"knn_model_11":knn_model_11, 
	# 					"knn_model_13":knn_model_13, 
	# 					'svm_linear':svm_linear, 
	# 					'svm_rbf':svm_rbf, 
	# 					#'gaussianprocess_model':gaussianprocess_model, 
	# 					'decisiontree_model':decisiontree_model, 
	# 					'rf_model_10':rf_model_10, 
	# 					'rf_model_100':rf_model_100, 
	# 					'rf_model_250':rf_model_250, 
	# 					'rf_model_500':rf_model_500, 
	# 					'rf_model_750':rf_model_750, 
	# 					'rf_model_1000':rf_model_1000, 
	# 					'mlp_model':mlp_model, 
	# 					'mlp_model_adaptive_learning':mlp_model_adaptive_learning, 
	# 					'adaboost_model':adaboost_model_10, 
	# 					"adaboost_model_50":adaboost_model_50, 
	# 					"adaboost_model_100":adaboost_model_100, 
	# 					"adaboost_model_150":adaboost_model_150, 
	# 					"adaboost_model_200":adaboost_model_200, 
	# 					"adaboost_model_250":adaboost_model_250, 
	# 					"adaboost_model_500":adaboost_model_500, 
	# 					"adaboost_model_750":adaboost_model_750, 
	# 					"adaboost_model_1000":adaboost_model_1000, 
	# 					"adaboost_model_2000":adaboost_model_2000, 
	# 					"adaboost_model_3000":adaboost_model_3000, 
	# 					# "adaboost_model_1000_2":adaboost_model_1000_2, 
	# 					# "adaboost_model_1000_5":adaboost_model_1000_5, 
	# 					# "adaboost_model_1000_10":adaboost_model_1000_10, 
	# 					'naive_bayes_model':naive_bayes_model, 
	# 					'quadrant_discriminant_analysis_model':quadrant_discriminant_analysis_model, 
	# 					#'linear_discriminant_analysis_model':linear_discriminant_analysis_model
	# 					# "Passive_Aggressive_Classifier":Passive_Aggressive_Classifier, 
	# 					"SGD_Classifier":SGD_Classifier, 
	# 					"Radius_Neighbours_classifier":Radius_Neighbours_classifier, 
	# 					"Ridge_classifier_cv":Ridge_classifier_cv, 
	# 					"Etree_Classifier":Etree_Classifier, 
	# 					"Nu_svc":Nu_svc
	# 					}
	# 	return(models_dict)

	#
	# def h2o_models():
	# 	import h2o
	# 	h2o.init()
	# 	from h2o.estimators import *
	# 	h2o_gbe = h2o.estimators.gbm.H2OGradientBoostingEstimator(model_id=None, 
	#                                                 distribution=None, 
	#                                                 quantile_alpha=None, 
	#                                                 tweedie_power=None, 
	#                                                 ntrees=100, 
	#                                                 max_depth=100, 
	#                                                 min_rows=None, 
	#                                                 learn_rate=0.01, 
	#                                                 nbins=None, 
	#                                                 sample_rate=0.9, 
	#                                                 col_sample_rate=0.3, 
	#                                                 col_sample_rate_per_tree=None, 
	#                                                 nbins_top_level=None, 
	#                                                 nbins_cats=None, 
	#                                                 balance_classes=None, 
	#                                                 max_after_balance_size=None, 
	#                                                 seed=1234, 
	#                                                 build_tree_one_node=None, 
	#                                                 nfolds=5, 
	#                                                 fold_assignment=None, 
	#                                                 keep_cross_validation_predictions=True, 
	#                                                 stopping_rounds=None, 
	#                                                 stopping_metric="misclassification", 
	#                                                 stopping_tolerance=None, 
	#                                                 score_each_iteration=None, 
	#                                                 score_tree_interval=None, 
	#                                                 checkpoint=None)
	#
	# 	models_dict = {'h2o_gbe':h2o_gbe, 
	# 	}
