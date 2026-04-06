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

TIKHONOV_REG_PARAMETER = [0.1, 0.01, 0.001]

path = '../input/'
df = pd.read_csv(path+'insurData.csv')
print('\nNumber of rows and columns in the data set: ', df.shape)
print('')


"""
    Returns a split variation of the dataset
    usage:
    X_train, X_test, y_train, y_test = split_df(df, test_size)
"""
def split_df(_df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = _df.drop(columns=[TARGET_COLUMN])
    y = _df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=test_size)


def preprocess(_df: pd.DataFrame):
    _df[TARGET_COLUMN] = _df[TARGET_COLUMN]/1000 #divide charges by 1000
    _df[SEX_COLUMN] = _df[SEX_COLUMN].apply(lambda x: 1 if x=="male" else 0)
    _df[SMOKER_COLUMN] = _df[SMOKER_COLUMN].apply(lambda x: 1 if x=="yes" else 0)
    _df = pd.get_dummies(_df, columns=[REGION_COLUMN], dtype=float) #Hot-encoding regions
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

print(df.head())

df = preprocess(df)
columns_names = list(df.columns)
X_train, X_test, y_train, y_test = split_df(df, TEST_SIZE)

#Lets look into top few rows and columns in the dataset
print(X_train.head())
print(y_train.head())
print(X_test.head())
print(y_test.head())

print(X_train.shape)
print(y_train.shape)
train_samples = y_train.shape
print(X_test.shape)
print(y_test.shape)
test_samples = y_test.shape

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print(X_train)

#### Single scalar model ####


#### Least squares model ####
for reg_param in TIKHONOV_REG_PARAMETER:
    _alpha_arr = tikhonov_regularized_solve(X_train, y_train, reg_param)
    y_pred = np.matmul(X_test, _alpha_arr)
    
    print(f"regularization parameter: {reg_param}")
    print(f"MSE: {((np.linalg.norm(y_test-y_pred))**2)/test_samples}")
    sanity_check(column_names=columns_names, y_test=y_test, y_pred=y_pred, _alpha_arr=_alpha_arr)