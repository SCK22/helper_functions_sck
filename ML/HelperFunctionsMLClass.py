"""Helper functions for Machine Learning"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, recall_score, f1_score

class HelperFunctionsML:
	"""Helper functions for Machine Learning and EDA"""

	def __init__(self, dataset):
		"""Helper Functions to do ML"""
		self.dataset = dataset
		self.target = None
		self.col_names = self.dataset.columns
		self.cat_num_extracted = False
		self.dummies = False
		self.nrows = dataset.shape[0]
		self.ncols = dataset.shape[1]
		self.actions_performed = {}
		self.model_columns = []
		self.model = None

	def update_actions_performed(self, dict_key, attributes = {}, needed_for_test = False):
		self.actions_performed[dict_key] = {"attributes" : attributes, "needed_for_test" : needed_for_test }
	# setter methods
	def set_target(self, col_name):
		self.target = col_name
		self.update_actions_performed("set_target", attributes =  {"col_name" : col_name})
		return "Succesfully set target column!"
	
	def test(self, col_name):
		temp = getattr(self, "set_target")(col_name)
		# print("getattr(self.set_target)(col_name) : {}".format(temp))

	def set_X_train(self, X_train):
		self.X_train = X_train
		self.update_actions_performed("set_X_train", attributes =  {"X_train" : X_train})
	
	def set_X_validation(self, X_validation):
		self.X_validation = X_validation
		self.update_actions_performed("set_X_validation", attributes =   {"X_validation" : X_validation})

	def set_y_train(self, y_train):
		self.y_train = y_train
		self.update_actions_performed("set_y_train", attributes =  {"y_train" : y_train})
	
	def set_y_validation(self, y_validation):
		self.y_validation = y_validation
		self.update_actions_performed("set_y_validation", attributes =  {"y_validation" : y_validation})

	def check_has_na_values(self):
		has_na_values = True if np.sum(self.dataset.isnull().sum())>0 else False
		self.has_na_values = has_na_values
		self.update_actions_performed("check_has_na_values")
		if self.has_na_values:
			print(self.dataset.isnull().sum())
		else:
			print("No Null values in the dataset.")
		return has_na_values
	
	# @staticmethod
	def cat_num_extract(self):
		f_name = "cat_num_extract"
		"""This function returns the names of the Categorical and Nmeric attributes in the same order."""
		self.cat_cols = [i for i in self.dataset.columns.values if self.dataset[i].dtype in ['O', 'object'] and i != self.target]
		self.num_cols = [i for i in self.dataset.columns.values if self.dataset[i].dtype in ['int', 'float'] and i != self.target]
		self.cat_num_extracted = True
		self.update_actions_performed(f_name)
		return {"cat_cols" : self.cat_cols, "num_cols": self.num_cols}
	
	def extend_cat_cols(self, new_cols = []):
		if not self.cat_num_extracted:
			print("Running cat_num_extract")
			self.cat_num_extract()
		self.cat_cols.extend(new_cols)

	def get_num_cols(self):
		return self.cat_num_extract()["num_cols"]
	def get_cat_cols(self):
		return self.cat_num_extract()["cat_cols"]
	def extend_num_cols(self, new_cols = []):
		if not self.cat_num_extracted:
			print("Running cat_num_extract")
			self.cat_num_extract()
		self.cat_cols.extend(new_cols)
		
	@staticmethod
	def list_of_na_cols(dataset, per=0.3):
		"""This function will return the columns with na values"""
		na_cols = dataset.isnull().sum()[dataset.isnull().sum() > per]
		return list(na_cols.index)

	def get_mode(self, x):
		values, counts = np.unique(x.dropna(), return_counts=True)
		m = counts.argmax()
		return values[m]
	
	def numeric_pipeline(self,strategy = "median"):
		numeric_transformer = Pipeline(steps=[
    	('imputer', SimpleImputer(strategy=strategy)),
    	('scaler', StandardScaler())])
		self.numeric_transformer = numeric_transformer

	def categorical_pipeline(self,strategy = "most_frequent"):
		categorical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy=strategy,fill_value="missing_value")),
    	('onehot', OneHotEncoder(handle_unknown='ignore'))])
		self.categorical_transformer = categorical_transformer

	def set_pipeline(self):
		self.preprocessor  = ColumnTransformer(
    	transformers=[
        ('num', self.numeric_transformer	, self.num_cols),
        ('cat', self.categorical_transformer, self.cat_cols)])
	
	def set_imputer_numeric(self, strategy = "median"):
		self.imputer_numeric = SimpleImputer(strategy = strategy)
		self.actions_performed["set_imputer_numeric"] = {"attributes" : {"strategy" : "median"}}

	def set_imputer_categorical(self, strategy = "most_frequent"):
		self.imputer_categorical = SimpleImputer(strategy = strategy)
		self.actions_performed["imputer_categorical"] = {"attributes" : {"strategy" : "most_frequent"}}
	
	def train_imputer_numeric(self):
		self.imputer_numeric.fit(self.dataset.loc[:,self.num_cols])
		self.update_actions_performed("train_imputer_numeric")
	
	def train_imputer_categorical(self):
		self.imputer_categorical.fit(self.dataset.loc[:,self.cat_cols])
		self.update_actions_performed("train_imputer_categorical")

	def impute_numeric_cols(self,test_dataset= None, return_frames = False):
		"""This function replaces the na values with the colum mean or median ,  based on the selection.
		This function might not be needed anymore as sklearn .22 has KNN imputation which I would prefer to use."""
		f_name = "impute_numeric_cols"
		num_cols = self.cat_num_extract()["num_cols"]
		print("num_cols : {}".format(num_cols))
		if test_dataset is not None:
			return self.imputer_numeric.transform(test_dataset.loc[:,num_cols])
		else:
			self.dataset.loc[:,num_cols] = self.imputer_numeric.transform(self.dataset.loc[:,num_cols])
		self.update_actions_performed(dict_key = f_name, needed_for_test = True)
		if return_frames:
			return self.dataset
	
	def impute_categorical_cols(self,test_dataset= None, return_frames = False):
		f_name = "impute_categorical_cols"
		print("cat_cols : {}".format(self.cat_cols))
		if test_dataset is not None:
			return self.imputer_categorical.transform(test_dataset.loc[:,self.cat_cols])
		else:
			self.dataset.loc[:,self.cat_cols] = self.imputer_categorical.transform(self.dataset.loc[:,self.cat_cols])
		self.update_actions_performed(dict_key = f_name, needed_for_test = True)
		if return_frames:
			return self.dataset

	def create_dummy_data_frame(self, categorical_attributes = None, return_frames = False):
		"""This function returns a dataframe of dummified colums Pass the dataset and the column names."""
		if categorical_attributes is None:
			categorical_attributes = self.cat_cols
			print("cat_cols : {}".format(categorical_attributes))
		if categorical_attributes:
			dummy_df =pd.get_dummies(self.dataset.loc[:,categorical_attributes]) 
			self.catergorical_dummies = dummy_df
			self.update_actions_performed(dict_key = "create_dummy_data_frame", attributes = {"columns_present" : list(dummy_df.columns)}, needed_for_test = True)
			self.dummies = True
			if return_frames:
				return dummy_df
		else:
			self.dummies = False
			if return_frames:
				return self.dataset

	def set_target_type(self, col_type):
		if self.target is not None:
			self.dataset.loc[:, [self.target]] = self.dataset.loc[:, [self.target]].astype(col_type)
		else:
			print("target not set, call the function set_target with the name of the target column")
	
	# def create_data_for_model(self):
	# 	if self.dummies:
	# 		self.model_data = pd.concat([self.dataset.loc[:, self.num_cols], self.catergorical_dummies, self.dataset.loc[:, self.target]], axis = 1)
	# 	else:
	# 		self.model_data = self.dataset
	
	def create_train_test_split(self, validation_size = 0.3, random_state= 42, return_frames= False):
		# self.create_data_for_model()
		self.model_data = self.dataset
		if self.target is not None:
			X = self.model_data.loc[:, [i for i in self.model_data.columns if i !=self.target]]
			y = self.model_data.loc[:, [self.target]]
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

	def compute_mertics(self, preds, trues):
		"""Compute various metrics"""
		try:
			acc = accuracy_score(y_pred=preds, y_true=trues)  # Accuracy for the passed set
		except: 
			acc = None
		try:
			prec = precision_score(y_pred=preds, y_true=trues, average='macro')  # precision for the passed set
		except: 
			prec = None
		try:
			rec = recall_score(y_pred=preds, y_true=trues, average='macro')  # recall for the passed set
		except:
			rec = None
		try:
			f1 = f1_score(y_pred=preds, y_true=trues)  # precision for the passed set
		except: 
			f1 = None

		model_performance = {}
		model_performance["precision"] = prec
		model_performance["f1_score"] = f1
		model_performance["recall"] = rec
		model_performance["accuracy"] = acc
		return model_performance	
	
	def apply_model_predict_validate(self, model_obj, model_name = None, feature_names= None):
		"""This function
		1.Applies the specified model to the train data.
		2.Validates on the validation set.
		3.Predicts on the test dataset
		4.Returns the predictions along with some scores.
		"""
		# if feature_names is None:
		# 	feature_names = []
		# 	feature_names = self.num_cols
		# 	feature_names.extend(self.cat_cols)
		# 	print("feature_names", feature_names)
		# 	if self.cat_num_extracted == False:
		# 		self.cat_num_extract()
		# 	self.model_columns.extend(self.cat_cols)
		# 	self.model_columns.extend(self.num_cols)
		# 	feature_names = self.model_columns
		# print("Using {} columns for model building".format(feature_names))
		# fit the model
		self.create_train_test_split()
		# self.numeric_pipeline()
		# self.categorical_pipeline()
		# self.set_pipeline()
		model_obj = Pipeline(steps=[('preprocessor', self.preprocessor),('classifier', model_obj)])
		self.model = model_obj
		self.model.fit(self.X_train, self.y_train)
	 	# evaluation metrics
		pred_train = self.model.predict(self.X_train)  # predict on the validation set 
		pred_val = self.model.predict(self.X_validation)  # predict on the validation set
		performance = {}
		performance["train"] = self.compute_mertics(pred_train, self.y_train)
		performance["validation"] = self.compute_mertics(pred_val, self.y_validation)
		if model_name is None:
			try:
				if "sklearn" in type(self.model).__module__:
					performance["model_obj"] = type(self.model).__name__
			except:
				performance["model_obj"] = self.model
		else:
			performance["model_obj"] = model_name
		# print("model_performance : {}".format(model_performance))	
		print(pd.DataFrame(performance))
		return(performance)

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

	# Experimental
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
