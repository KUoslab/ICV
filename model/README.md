This project experiments with predicting the CPU quota requirements of the Linux virtual machine I/O thread vHost using machine learning.

## Development Environment
* Python 3.8
* Pycharm Community Edition 2020.2

## Machine Learning Concepts
The following three supervised regression algorithms are used:

- Linear Regression Model (LR)
- Support Vector Machine Regression (SVM)
- Random Forest Regression (RF)

For each model, its input values represent the packet size, TX bandwidth, TX pps, and the vCPU usage (located underneath the *data* folder in CVS format), while its predicted values as well as its target values represent the optimal CPU quota value. Either the Root Mean Square Logarithmic Error (RMLSE) or just the Root Mean Square Error (RMSE) can be selected as the evaluation metric. 

## Getting Started
1. Install the following Python packages: openpyxl, numpy, pandas, scikit-learn, joblib.
2. Navigate to the folder that includes the desired evaluation metric in its name. For example, if RMLSE is desired, follow the rest of this guideline with the files underneath the folder *cpu_quota_rmsle*.
3. Open the Python file that has the desired supervised regression algorithm as its name. For example, if Linear Regression is desired, open the Python file *linear.py*.

## Training and Cross-validating a Model
(Validation is only available for SVM and RF).

1. Uncomment the *Cross validate* and *Save model* sections.
2. Comment the *Test*, *Evaluate*, and *Save* sections.
3. Execute the Python file.

The cross-validated model will be saved underneath the folder *model* with the name set as the value of the variable *model_name*.

## Testing a Saved(Validated) Model

1. Comment the *Cross validate* and *Save model* sections.
2. Uncomment the *Test*, *Evaluate*, and *Save* sections.
3. Execute the Python file.

Results will be saved in the path given as the argument to *wb.save()*, located at the end of the Python file.

## References

### Keras
- https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

### MLP
- https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/

### Supervised Regression Algorithms

**Random Forest**
- https://towardsdatascience.com/understanding-random-forest-58381e0602d2
- https://towardsdatascience.com/random-forest-and-its-implementation-71824ced454f
- https://heartbeat.fritz.ai/random-forest-regression-in-python-using-scikit-learn-9e9b147e2153 (code)
- https://towardsdatascience.com/random-forest-in-python-24d0893d51c0 (code)
- https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/ (code)

**Linear Regression**
- https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html (code)

**Support Vector Regression**
- https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html (code)

### Python Libraries

**Import**
- https://www.excelforum.com/excel-general/492007-select-every-other-row-to-copy-cut.html
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

**Split**
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

**Normalize**
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
- https://stackoverflow.com/questions/34165731/a-column-vector-y-was-passed-when-a-1d-array-was-expected
- https://datascience.stackexchange.com/questions/12321/whats-the-difference-between-fit-and-fit-transform-in-scikit-learn-models

**Train**
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=random%20forest#sklearn.ensemble.RandomForestRegressor (RF)
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression (LR)
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR (SVR)

**Evaluate (RMSE)**
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

**Save**
- https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
