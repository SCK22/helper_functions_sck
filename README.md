# HelperFunctions for EDA and model building in python

Under active development!

> Supports structured data as of now

Do you work on data analysis and model building projects on a daily basis?

Did you notice the number of lines of code that are duplicated across your analyses?

### This is a collection of functions that can be used for analysis and model building. It follows the convention of creating an object instance of the class HelperFunctionsML, upon which we can apply a set of functions/actions.

Almost every action on the dataset is applied `inplace` , i.e the changed/updated dataset is updated and this new dataset is used for the next function call that is applied.
or simply put, the class treats the dataset as a shared variable across various functions.
Allowing for a cleaner code.

#### To convert the dataset into a HelperFunctionsML object:

```
obj = HelperFunctionsML(pd.read_csv("dataset.csv"))
```

**Once the dataset is converted into the HelperFunctionsML object, we have access to some metadata and some useful functions**
Some of the functions available are"

```
cat_num_extract - This function returns the names of the Categorical and Nmeric attributes.
list_of_na_cols
impute_categorical_cols
impute_numeric_cols
create_dummy_data_frame
```

Interested in using these functions? checkout my [ipython notebook](howtouse/Modeling with Titanic Data using HelperFunctions.ipynb) which outlines how these functions can be used.
Titanic dataset, it is a simple and small dataset which allows us to concentrate more on how to use this module.
Suggestions are welcome, please click here to send an email: [S.Chaithanya Kumar](mailto:chaithanyakumar.ds@gmail.com?subject=[GitHub]SCK22/HelperFunctions)

# To Do

Support for using the models on test data coming soon. Stay tuned!

# Attributions

Thanks to https://github.com/ternaus for the setup.py file. You can find the file he wrote here[https://github.com/ternaus/retinaface/blob/master/setup.py]
