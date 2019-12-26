"""Helper functions for Machine Learning"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.svm import SVC, NuSVC
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.ensemble import RandomForestClassifier,  AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.neighbors import NearestCentroid
# from sklearn.linear_model import RidgeClassifier
# from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, SGDClassifier, PassiveAggressiveClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

class HelperFunctionsML:
	"""Helper functions for Machine Learning and EDA"""

	def __init__(self, dataset):
		"""Helper Functions to do ML"""
		self.dataset = dataset
		self.target = None
		self.col_names = self.dataset.columns
		self.nrows = dataset.shape[0]
		self.ncols = dataset.shape[1]
	# setter methods
	def set_target(self, col_name):
		self.target = col_name
	
	def set_X_train(self, X_train):
		self.X_train = X_train
	
	def set_X_validation(self, X_validation):
		self.X_validation = X_validation
	def set_y_train(self, y_train):
		self.y_train = y_train
	def set_y_train(self, y_train):
		self.y_train  = y_train
	def set_y_validation(self, y_validation):
		self.y_validation = y_validation

	def check_has_na_values(self):
		has_na_values = True if np.sum(self.dataset.isnull().sum())>0 else False
		self.has_na_values = has_na_values
		return has_na_values
	
	# @staticmethod
	def cat_num_extract(self):
		"""This function returns the names of the Categorical and Nmeric attributes in the same order."""
		cat_cols = [i for i in self.dataset.columns.values if self.dataset[i].dtype in ['O', 'object']]
		num_cols = [i for i in self.dataset.columns.values if self.dataset[i].dtype not in ['O', 'object']]
		return {"cat_cols" : cat_cols, "num_cols": num_cols}

	@staticmethod
	def list_of_na_cols(dataset, per=0.3):
		"""This function will return the columns with na values"""
		na_cols = dataset.isnull().sum()[dataset.isnull().sum() > per]
		return list(na_cols.index)

	def get_mode(self, x):
		values, counts = np.unique(x.dropna(), return_counts=True)
		m = counts.argmax()
		return values[m]

	def impute_categorical_cols(self, return_frames = False):
		"""This function replaces the na values with the colum mode
		This function might not be needed anymore as sklearn .22 has KNN imputation which I would prefer to use."""
		cat_cols = self.cat_num_extract()["cat_cols"]
		if cat_cols:
			self.dataset.loc[:,cat_cols].isnull
			self.dataset.loc[:,cat_cols] = self.dataset.loc[:,cat_cols].apply(lambda x : x.fillna(value=self.get_mode(x)))
		if return_frames:
			return self.dataset

	def impute_numeric_cols(self, method="median", return_frames = False):
		"""This function replaces the na values with the colum mean or median ,  based on the selection.
		Available values for method are "meidan",  "mean". default is median
		This function might not be needed anymore as sklearn .22 has KNN imputation which I would prefer to use."""
		num_cols = self.cat_num_extract()["num_cols"]
		if num_cols and method == "mean":
			self.dataset = self.dataset.loc[:,num_cols].apply(lambda x : x.fillna(value=x.mean()))
		if num_cols and method == "median":
			self.dataset = self.dataset.loc[:,num_cols].apply(lambda x : x.fillna(value=x.median()))
		if return_frames:
			return self.dataset
	
	def create_dummy_data_frame(self, categorical_attributes = None):
		"""This function returns a dataframe of dummified colums Pass the dataset and the column names."""
		if categorical_attributes is None:
			categorical_attributes = self.cat_num_extract()["cat_cols"]
			print("cat_cols : {}".format(categorical_attributes))
		if categorical_attributes:
			return (pd.get_dummies(self.dataset[categorical_attributes]))
		else:
			return self.dataset
	

	def set_target_type(self, col_type):
		if self.target is not None:
			self.dataset.loc[:, [self.target]] = self.dataset.loc[:, [self.target]].astype(col_type)
		else:
			print("target not set, call the function set_target with the name of the target column")

	def create_train_test_split(self, validation_size = 0.3, random_state= 42, return_frames= False):

		if self.target is not None:
			X = self.dataset.loc[:, [i for i in self.col_names if i !=self.target]]
			y = self.dataset.loc[:, [self.target]]
			X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size = validation_size, random_state= random_state) 
			self.X_train = X_train
			self.X_validation = X_validation
			self.y_train  = y_train
			self.y_validation = y_validation
			self.nrowstrain, self.ncolstrain = X_train.shape
			self.nrowvalidation, self.ncolsvalidation = X_validation.shape
		else:
			print("target not set, call the function set_target with the name of the target column")
		if return_frames:
			return X_train, X_validation, y_train, y_validation

	def compute_mertics(self, pred_val):
		"""Compute various metrics"""
		try:
			acc_val = accuracy_score(y_pred=pred_val, y_true=self.y_validation)  # Accuracy for the validation set
		except: 
			acc_val = None
		try:
			prec_val = precision_score(y_pred=pred_val, y_true=self.y_validation, average='macro')  # precision for the validation set
		except: 
			prec_val = None
		try:
			rec_val = recall_score(y_pred=pred_val, y_true=self.y_validation, average='macro')  # recall for the validation set
		except:
			rec_val = None
		try:
			f1_val = f1_score(y_pred=pred_val, y_true=self.y_validation)  # precision for the validation set
		except: 
			f1_val = None

		model_performance = {}
		model_performance["precision"] = prec_val
		model_performance["f1_score"] = f1_val
		model_performance["recall"] = rec_val
		model_performance["accuracy"] = acc_val
		return model_performance
		

	def apply_model_predict_validate(self, model_obj, model_name = None, feature_names= None):
		"""This function
		1.Applies the specified model to the train data.
		2.Validates on the validation set.
		3.Predicts on the test dataset
		4.Returns the predictions along with some scores.
		"""
		if feature_names is None:
			feature_names = self.X_train.columns
		# apply the model
		
		model_obj.fit(self.X_train.loc[:, feature_names], self.y_train)
	 	# evaluation metrics
		pred_val = model_obj.predict(self.X_validation[feature_names])  # predict on the validation set
		
		model_performance = self.compute_mertics(pred_val)
		if model_name is None:
			try:
				if "sklearn" in type(model_obj).__module__:
					model_performance["model_obj"] = type(model_obj).__name__
			except:
				model_performance["model_obj"] = model_obj
		else:
			model_performance["model_obj"] = model_name
		# print("model_performance : {}".format(model_performance))
		return(model_performance)


	def apply_log_reg(self, feature_names = None):
		"""apply basic logistic regression model"""
		model = LogisticRegression()
		return self.apply_model_predict_validate(model, feature_names = feature_names)

	def apply_dtree_class(self, feature_names = None):
		"""apply basic Decision tree classification model"""
		model = DecisionTreeClassifier()
		return self.apply_model_predict_validate(model, feature_names = feature_names)

	def compare_model_performance(self, model_objs_list = [], model_names_list = [], feature_names = None):
		"""pass a list of model objects, this function will apply all the models and return a dataframe with the performance 
		measures
		Input : list of model objects
				list of model names (optional)
				features to be used for model building
		"""
		if len(model_objs_list) == 1:
			return self.apply_model_predict_validate(model_objs_list[0], feature_names = feature_names)
		if len(model_objs_list) > 1:
			model_perf_list = []
			if model_names_list:
				for model, name in zip(model_objs_list,model_names_list):
					model_perf = self.apply_model_predict_validate(model, model_name=name, feature_names = feature_names)
					model_perf_list.append(model_perf)
			else:
				for model in model_objs_list:
					model_perf = self.apply_model_predict_validate(model, feature_names = feature_names)
					model_perf_list.append(model_perf)
		df = pd.DataFrame(model_perf_list)
		# df = df.set_index(df["model_obj"])
		# df.reset_index()
		return df





	# Replace values in an attribute with other values
	def replace_attribute_values(self, target_attribute, originals, replace_with):
		"""Experimental:
		 This function takes a pandas series object and replaces the specified values with specified values."""
		if len(originals) == len(replace_with):
			for i in range(len(originals)):
				self.X_train[target_attribute].replace(originals[i], replace_with[i], inplace=True)
				self.X_validation[target_attribute].replace(originals[i], replace_with[i], inplace=True)
		elif len(originals != len(replace_with)):
			raise ValueError("replacement values do not match the size of originals")
		return 

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
