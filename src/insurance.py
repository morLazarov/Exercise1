from typing import Tuple

import pandas as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization

from sklearn.model_selection import train_test_split

TARGET_COLUMN = "charges"
SEX_COLUMN = "sex"
SMOKER_COLUMN = "smoker"
REGION_COLUMN = "region"

TEST_SIZE = 0.2

TIKHONOV_REG_PARAMETER = 0.01

EXPERIMENT_NUMBER = "experiment_number"
DATASET_VERSION = "dataset_version"
MODEL_TYPE = "model_type"
TRAIN_MSE = "train_mse"
TEST_MSE = "test_mse"
RATIO_TO_SCALAR_MODEL = "ratio_to_scalar_model"

SCALAR_MODEL_TYPE = "scalar_model"
LS_MODEL_TYPE = "ls_model"

LS_MODEL_TRAINING_ITERATIONS = 10

path = '../input/'
df = pd.read_csv(path+'insurData.csv')
print('\nNumber of rows and columns in the data set: ', df.shape)
print('')

results_df = pd.DataFrame({EXPERIMENT_NUMBER: [], DATASET_VERSION: [], MODEL_TYPE: [], TRAIN_MSE: [], TEST_MSE: [], RATIO_TO_SCALAR_MODEL: []})


"""
    Returns a split variation of the dataset
    usage:
    X, y, X_train, X_test, y_train, y_test = split_df(df, test_size)
"""
def split_df(_df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = _df.drop(columns=[TARGET_COLUMN])
    y = _df[TARGET_COLUMN]
    return X, y, *train_test_split(X, y, test_size=test_size)


def preprocess1(_base_df: pd.DataFrame):
    _df = _base_df.copy()
    _df[TARGET_COLUMN] = _df[TARGET_COLUMN]/1000 #divide charges by 1000
    _df[SEX_COLUMN] = _df[SEX_COLUMN].apply(lambda x: 1 if x=="male" else 0)
    _df[SMOKER_COLUMN] = _df[SMOKER_COLUMN].apply(lambda x: 1 if x=="yes" else 0)
    _df = pd.get_dummies(_df, columns=[REGION_COLUMN], dtype=float) #Hot-encoding regions
    _df = _df.assign(a0 = 1) #Add a column of 1's to a copy of the dataset

    return _df

def preprocess2(_base_df: pd.DataFrame):
    _df = _base_df.copy()
    _df[TARGET_COLUMN] = _df[TARGET_COLUMN]/1000 #divide charges by 1000
    _df[SEX_COLUMN] = _df[SEX_COLUMN].apply(lambda x: 1 if x=="male" else 0)
    _df = _df.assign(a0 = 1) #Add a column of 1's to a copy of the dataset

    return _df

#Solves LS for a given matrix, b vector and alpha parameter and returns the minimizing vector
def tikhonov_regularized_solve(A: np.ndarray, b: np.ndarray, _reg_param: float) -> np.ndarray: 
    col_dim = A.shape[1]
    a_identity = _reg_param**2 * np.eye(col_dim)
    AT = np.matrix_transpose(A)
    ATb = np.matmul(AT, b)
    ATA = np.matmul(AT, A)
    x = np.linalg.solve(ATA + a_identity, ATb)
    return x


# TODO: remove this
def plot_pred_vs_gt(y_true, y_pred, title="Predicted vs Ground Truth"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    plt.figure()
    
    # Scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6)
    
    # Perfect prediction line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    
    # Labels and title
    plt.xlabel("Ground Truth (y)")
    plt.ylabel("Predicted (ŷ)")
    plt.title(title)
    
    # Optional: show correlation
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(0.05, 0.95, f"Corr = {corr:.3f}", transform=plt.gca().transAxes,
             verticalalignment='top')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plots a spread plot, and prints the contribution of each feature to the final predicted value
def sanity_check(column_names, y_test, y_pred, _alpha_arr):
    print(f"column_names: {str(column_names)}")
    print(f"alpha_arr: {_alpha_arr}")
    plot_pred_vs_gt(y_test, y_pred)

def calculate_MSE(gt, pred):
    return np.square(np.subtract(gt, pred)).mean()

def record_experiment(experiment_number, dataset_version, model_type, train_mse, test_mse, ratio_to_scalar_model):
    results_df.loc[len(results_df)] = [experiment_number, dataset_version, model_type, train_mse, test_mse, ratio_to_scalar_model]

def evaluation_pipeline(_df: pd.DataFrame, dataset_version: str):
    X, y, X_train, X_test, y_train_gt, y_test_gt = split_df(_df, TEST_SIZE)

    samples_count = len(y)

    X = X.to_numpy()
    y = y.to_numpy()
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train_gt = y_train_gt.to_numpy()
    y_test_gt = y_test_gt.to_numpy()


    # Plot histogram
    plt.figure()
    plt.hist(y, bins=30, density=True)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Distribution of values")
    plt.show()

    print(X_train)


    ### Single Scalar Model ###
    min_arg = (1/samples_count) * np.sum(y)
    a_0 = np.full(samples_count, min_arg)
    mse_0 = calculate_MSE(y, a_0)
    print(f"Scalar model MSE: {mse_0}")

    record_experiment(1, dataset_version, SCALAR_MODEL_TYPE, mse_0, mse_0, 1)

    for i in range(1, LS_MODEL_TRAINING_ITERATIONS+1):
        #### Least squares model ####
        print(f"regularization parameter: {TIKHONOV_REG_PARAMETER}")

        _alpha_arr = tikhonov_regularized_solve(X_train, y_train_gt, TIKHONOV_REG_PARAMETER)

        y_train_pred = np.matmul(X_train, _alpha_arr) #final predicted y values
        train_mse = calculate_MSE(y_train_gt, y_train_pred)
        print(f"train MSE: {train_mse}")

        y_test_pred = np.matmul(X_test, _alpha_arr) #final predicted y values
        test_mse = calculate_MSE(y_test_gt, y_test_pred)
        print(f"test MSE: {test_mse}")

        print(f"learned weights: {_alpha_arr}")

        record_experiment(i, dataset_version, LS_MODEL_TYPE, train_mse, test_mse, test_mse / mse_0)


preprocessed_df = preprocess1(df)
print(preprocessed_df.columns)
print("#### evaluation with smoking and regions")
evaluation_pipeline(preprocessed_df, "with_smoker_and_region")

preprocessed_df2 = preprocess2(df.drop(columns=[SMOKER_COLUMN, REGION_COLUMN]))
print(preprocessed_df2.columns)
print("#### evaluation without smoking and regions")
evaluation_pipeline(preprocessed_df2, "no_smoker_and_region")

print(results_df)