# Description: Utility functions for the project

import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def reshape_X(
    X_train: np.ndarray or pd.DataFrame, X_test: np.ndarray or pd.DataFrame
) -> tuple:
    """
    The function `reshape_X` takes in two inputs, `X_train` and `X_test`, which can be either a numpy
    array or a pandas DataFrame, and reshapes them into a 2-dimensional array if they are 1-dimensional.

    :param X_train: The training data, which can be either a numpy array or a pandas DataFrame. It
    represents the input features for training a machine learning model
    :type X_train: np.ndarray or pd.DataFrame
    :param X_test: The X_test parameter is the test dataset that you want to reshape. It can be either a
    numpy array or a pandas DataFrame
    :type X_test: np.ndarray or pd.DataFrame
    :return: a tuple containing the reshaped X_train and X_test arrays or dataframes.
    """
    if isinstance(X_train, (pd.DataFrame, pd.Series)) and X_train.values.ndim == 1:
        X_train = X_train.values.reshape(-1, 1)
    if isinstance(X_test, (pd.DataFrame, pd.Series)) and X_test.values.ndim == 1:
        X_test = X_test.values.reshape(-1, 1)
    if isinstance(X_train, np.ndarray) and X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if isinstance(X_test, np.ndarray) and X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    return X_train, X_test


def train_test_split_01(data_df: pd.DataFrame, train_size: float = 0.8) -> tuple:
    """
    The function `train_test_split_01` takes a pandas DataFrame and a train size as input, and returns a
    tuple containing the train data and test data based on the specified train size.

    :param data_df: The data_df parameter is a pandas DataFrame that contains the data you want to split
    into training and testing sets
    :type data_df: pd.DataFrame
    :param train_size: The train_size parameter is the proportion of the data that should be used for
    training. It is a float value between 0 and 1, where 0 represents 0% of the data and 1 represents
    100% of the data. By default, the train_size is set to
    :type train_size: float
    :return: The function train_test_split_01 returns a tuple containing the train_data and test_data.
    """
    data_df = data_df.copy()
    train_index = int(len(data_df) * train_size)
    train_data = data_df.iloc[:train_index]
    test_data = data_df.iloc[train_index:]
    return train_data, test_data


def evaluate_model(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    The function evaluates the performance of a regression model by calculating and printing various
    error metrics.

    :param y_test: The true values of the target variable (dependent variable) from the test dataset
    :type y_test: np.ndarray
    :param y_pred: The predicted values of the target variable
    :type y_pred: np.ndarray
    """
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("R-squared (R2) Score:", r2)


def plot_train_test_split(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """
    The function `plot_train_test_split` plots the training and test data sets on the same graph, with a
    vertical line indicating the start of the test data.

    :param train_data: A pandas DataFrame containing the training data
    :type train_data: pd.DataFrame
    :param test_data: The `test_data` parameter is a pandas DataFrame that contains the data for the
    test set
    :type test_data: pd.DataFrame
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    train_data.plot(ax=ax, label="Training Set", title="Data Train/Test Split")
    test_data.plot(ax=ax, label="Test Set")
    first_test_date = test_data.iloc[0].name
    ax.axvline(first_test_date, color="black", ls="--")
    ax.legend(["Training Set", "Test Set"])
    plt.show()


def polynomial_regression(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, max_degree: int = 10
) -> np.ndarray:
    """
    The `polynomial_regression` function performs polynomial regression on the given training data and
    predicts the target variable for the given test data using the best degree of polynomial determined
    through grid search.

    :param X_train: X_train is a numpy array containing the training data for the independent variables
    in polynomial regression. Each row represents a data point, and each column represents a different
    independent variable
    :type X_train: np.ndarray
    :param y_train: The `y_train` parameter represents the target variable or the dependent variable in
    the training dataset. It is a numpy array that contains the actual values of the target variable
    corresponding to each row in the `X_train` dataset
    :type y_train: np.ndarray
    :param X_test: X_test is a numpy array containing the input features for which we want to make
    predictions using the polynomial regression model
    :type X_test: np.ndarray
    :param max_degree: The `max_degree` parameter specifies the maximum degree of the polynomial
    features to consider in the polynomial regression model. It determines the complexity of the model
    and the number of features that will be generated from the input data, defaults to 10
    :type max_degree: int (optional)
    :return: The function `polynomial_regression` returns the predicted values for the input `X_test`
    using a polynomial regression model.
    """

    def get_best_degree(
        _X_train: np.ndarray, _y_train: np.ndarray, _max_degree: int
    ) -> int:
        """
        The function `get_best_degree` uses grid search to find the best degree for polynomial regression
        based on the given training data.

        :param _X_train: The `_X_train` parameter is a numpy array that represents the training data for the
        independent variables (features) in a machine learning model
        :type _X_train: np.ndarray
        :param _y_train: The parameter `_y_train` represents the target variable or the dependent variable
        in your training dataset. It is a numpy array containing the values of the target variable for each
        observation in your training data
        :type _y_train: np.ndarray
        :param _max_degree: The `_max_degree` parameter represents the maximum degree of polynomial features
        to consider when fitting the model
        :type _max_degree: int
        :return: the best degree of polynomial features to use in a linear regression model.
        """
        param_grid = {"polynomialfeatures__degree": list(range(1, _max_degree + 1))}
        model = Pipeline(
            [
                ("polynomialfeatures", PolynomialFeatures()),
                ("linearregression", LinearRegression()),
            ]
        )

        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
        grid_search.fit(_X_train, _y_train)
        return grid_search.best_params_["polynomialfeatures__degree"]

    best_degree = get_best_degree(X_train, y_train, max_degree)

    # Fit polynomial regression model with the best degree
    model = Pipeline(
        [
            ("polynomialfeatures", PolynomialFeatures(degree=best_degree)),
            ("linearregression", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def get_y_pred(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    regression_type: str = "lr",
) -> np.ndarray:
    """
    The function `get_y_pred` takes in training and testing data, along with a regression type, and
    returns the predicted values for the testing data using linear regression or polynomial regression.

    :param X_train: A pandas DataFrame containing the features (input variables) for training the
    regression model
    :type X_train: pd.DataFrame
    :param y_train: The `y_train` parameter is a pandas DataFrame that contains the target variable
    values for the training data
    :type y_train: pd.DataFrame
    :param X_test: A DataFrame containing the features of the test data
    :type X_test: pd.DataFrame
    :param regression_type: The `regression_type` parameter is a string that specifies the type of
    regression to be performed. It can take two possible values:, defaults to lr
    :type regression_type: str (optional)
    :return: a numpy array containing the predicted values.
    """
    X_train, X_test = reshape_X(X_train, X_test)
    regressor = LinearRegression()

    if regression_type == "poly":
        return polynomial_regression(X_train, y_train, X_test, max_degree=10)

    model = regressor.fit(X_train, y_train)
    return model.predict(X_test)


def train_test_split_02(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    X_column_name: str,
    y_column_name: str,
) -> tuple:
    """
    The function `train_test_split_02` takes in train and test data, extracts the specified X and y
    columns, converts the X columns to timestamps, and returns the X and y values for both the train and
    test data.

    :param train_data: The `train_data` parameter is a pandas DataFrame that contains the training data.
    It should have columns for the features (X) and the target variable (y)
    :type train_data: pd.DataFrame
    :param test_data: The `test_data` parameter is a pandas DataFrame that contains the data that you
    want to use for testing your model. It should have the same columns as the `train_data` DataFrame
    :type test_data: pd.DataFrame
    :param X_column_name: The `X_column_name` parameter is the name of the column in the data that
    contains the features or independent variables. These are the variables that will be used to predict
    the target variable
    :type X_column_name: str
    :param y_column_name: The `y_column_name` parameter is the name of the column in the data that
    contains the target variable or the variable you want to predict
    :type y_column_name: str
    :return: The function `train_test_split_02` returns a tuple containing four elements: `X_train`,
    `y_train`, `X_test`, and `y_test`.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    if train_data.index.name:
        train_data.reset_index(inplace=True)

    if test_data.index.name:
        test_data.reset_index(inplace=True)

    X_train = train_data[X_column_name].apply(lambda x: x.timestamp())
    y_train = train_data[y_column_name]
    X_test = test_data[X_column_name].apply(lambda x: x.timestamp())
    y_test = test_data[y_column_name]
    return X_train, y_train, X_test, y_test


def get_regression_data(
    data_df: pd.DataFrame,
    X_column_name: str,
    y_column_name: str,
    regression_type: str = "lr",
    convert_to_df: bool = True,
) -> pd.DataFrame:
    """
    The function `get_regression_data` takes a DataFrame, splits it into train and test data, performs
    regression on the train data, and returns the predicted values for both train and test data.

    :param data_df: The `data_df` parameter is a pandas DataFrame that contains the data for regression
    analysis. It should have columns for both the independent variable (`X_column_name`) and the
    dependent variable (`y_column_name`)
    :type data_df: pd.DataFrame
    :param X_column_name: The name of the column in the data DataFrame that contains the independent
    variable (X variable) for regression analysis
    :type X_column_name: str
    :param y_column_name: The `y_column_name` parameter is the name of the column in the `data_df`
    DataFrame that contains the target variable (the variable you want to predict)
    :type y_column_name: str
    :param regression_type: The `regression_type` parameter is used to specify the type of regression
    algorithm to use. The default value is "lr", which stands for linear regression. This means that by
    default, the function will use linear regression to predict the target variable. However, you can
    also specify other regression algorithms such, defaults to lr
    :type regression_type: str (optional)
    :param convert_to_df: The `convert_to_df` parameter is a boolean flag that determines whether the
    output should be converted to a pandas DataFrame or not. If `convert_to_df` is set to `True`, the
    output will be a pandas DataFrame with the regression data. If `convert_to_df` is set to `, defaults
    to True
    :type convert_to_df: bool (optional)
    :return: either a pandas DataFrame or a numpy array, depending on the value of the `convert_to_df`
    parameter. If `convert_to_df` is `True`, a DataFrame is returned with the regression data, where the
    columns are named based on the `y_column_name` parameter and the index is either the original index
    of the `data_df` DataFrame or the values from the `
    """

    data_df = data_df.copy()
    train_data, test_data = train_test_split_01(data_df)
    X_train, y_train, X_test, y_test = train_test_split_02(
        train_data, test_data, X_column_name, y_column_name
    )

    y_train_regression = get_y_pred(X_train, y_train, X_train, regression_type)
    y_test_regression = get_y_pred(X_train, y_train, X_test, regression_type)
    y_regression_data = np.concatenate((y_train_regression, y_test_regression))

    if convert_to_df:
        idx = data_df.index if data_df.index.name else data_df[X_column_name]
        return pd.DataFrame(y_regression_data, columns=[y_column_name], index=idx)
    return y_regression_data


def plot_all_data_with_entire_regression_line(
    data_df: pd.DataFrame,
    X_column_name: str,
    y_column_name: str,
    regression_type: str = "lr",
    eval_model: bool = False,
) -> None:
    """
    The function plots the actual data and the regression line for a given dataset, and optionally
    evaluates the performance of the regression model.

    :param data_df: data_df is a pandas DataFrame that contains the unsplit training and testing data.
    It should have columns for the independent variable (X) and the dependent variable (y)
    :type data_df: pd.DataFrame
    :param X_column_name: The `X_column_name` parameter is the name of the column in the `data_df`
    DataFrame that contains the independent variable (input variable) data. This column will be used as
    the input for the regression model
    :type X_column_name: str
    :param y_column_name: The `y_column_name` parameter is the name of the column in the `data_df`
    DataFrame that contains the target variable or the variable you want to predict
    :type y_column_name: str
    :param regression_type: The `regression_type` parameter specifies the type of regression to be used.
    It has a default value of "lr", which stands for linear regression, defaults to lr
    :type regression_type: str (optional)
    :param eval_model: The `eval_model` parameter is a boolean flag that determines whether or not to
    evaluate the performance of the regression model. If `eval_model` is set to `True`, the function
    will evaluate the model by calculating and displaying various performance metrics such as mean
    squared error, mean absolute error, and R, defaults to False
    :type eval_model: bool (optional)
    """

    data_df = data_df.copy()

    regression_df = get_regression_data(
        data_df,
        X_column_name,
        y_column_name,
        regression_type=regression_type,
        convert_to_df=True,
    )
    data_df = data_df[y_column_name].to_frame()

    fig, ax = plt.subplots(figsize=(15, 5))
    data_df.plot(ax=ax, label="Actual", title="Actual vs Linear Regression")
    regression_df.plot(ax=ax, label="Regression Training Data")

    train_data, test_data = train_test_split_01(data_df)
    first_test_date = test_data.iloc[0].name
    ax.axvline(first_test_date, color="black", ls="--")
    ax.legend(["Actual", f"{regression_type.upper()} Predicted"])
    plt.show()

    if eval_model:
        X_train, y_train, X_test, y_test = train_test_split_02(
            train_data, test_data, X_column_name, y_column_name
        )
        y_pred = get_y_pred(X_train, y_train, X_test, regression_type)
        evaluate_model(y_test, y_pred)


def plot_data_with_regression_and_prediction(
    reg_train_data: np.ndarray, y_pred: np.ndarray, data_df: pd.DataFrame
) -> None:
    """
    The function `plot_data_with_regression_and_prediction` plots the actual data, linear regression
    training data, and linear regression prediction on a graph.

    :param reg_train_data: The `reg_train_data` parameter is a numpy array that contains the training
    data used for linear regression. It should have shape `(n_samples, n_features)`, where `n_samples`
    is the number of samples and `n_features` is the number of features in the training data
    :type reg_train_data: np.ndarray
    :param y_pred: The `y_pred` parameter is a numpy array that contains the predicted values for the
    target variable (in this case, the "Balance" variable) based on a linear regression model
    :type y_pred: np.ndarray
    :param data_df: `data_df` is a pandas DataFrame that contains the actual data that you want to plot.
    It should have a column named "Balance" which represents the values you want to plot
    :type data_df: pd.DataFrame
    """
    train_index = data_df.index[len(data_df) - len(reg_train_data) :]
    reg_train_df = pd.DataFrame(reg_train_data, columns=["Balance"], index=train_index)
    y_pred_df = pd.DataFrame(y_pred, columns=["Balance"], index=train_index)

    fig, ax = plt.subplots(figsize=(15, 5))
    data_df.plot(
        ax=ax,
        label="Actual",
        title="Actual vs Linear Regression Training Data vs Linear Regression Prediction",
    )
    reg_train_df.plot(ax=ax, label="Linear Regression Training Data")
    y_pred_df.plot(ax=ax, label="Linear Regression Prediction")
    ax.legend(["Actual", "Predicted"])
    ax.axvline(train_index[0], color="black", ls="--")
    plt.show()


def graph_checkings(df: pd.DataFrame) -> None:
    """
    The function `graph_checkings` creates a scatter plot to visualize the balance over time in a
    DataFrame.

    :param df: The parameter `df` is a pandas DataFrame that contains the data for the graph. It is
    expected to have a column named "Balance" which represents the balance values over time
    :type df: pd.DataFrame
    """
    df = df.copy()
    x = df.index.values
    y = df["Balance"].values

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.scatter(x, y, c="g", alpha=0.7, s=25, label="Balance over time")
    ax.set_title("Balance Over Time", fontsize="xx-large", fontweight="bold")
    ax.grid(True)
    fig.autofmt_xdate()
    plt.show()


def X_y_split(
    train_data: pd.DataFrame, X_column_name: str, y_column_name: str
) -> tuple:
    """
    The function `X_y_split` takes a pandas DataFrame, extracts the specified X and y columns, converts
    the X column values to timestamps, and returns X and y as separate arrays.

    :param train_data: A pandas DataFrame containing the training data
    :type train_data: pd.DataFrame
    :param X_column_name: The X_column_name parameter is the name of the column in the train_data
    DataFrame that contains the features or independent variables
    :type X_column_name: str
    :param y_column_name: The `y_column_name` parameter is the name of the column in the `train_data`
    DataFrame that contains the target variable or the variable you want to predict
    :type y_column_name: str
    :return: a tuple containing two values: X and y.
    """
    train_data = train_data.copy()

    if train_data.index.name:
        train_data.reset_index(inplace=True)

    X = train_data[X_column_name].apply(lambda x: x.timestamp())
    X = X.values.reshape(-1, 1)
    y = train_data[y_column_name].values
    return X, y


def plot_train_data_with_regression(
    train_df: pd.DataFrame,
    X_column_name: str,
    y_column_name: str,
    regression_type: str = "lr",
    eval_model: bool = False,
) -> None:
    """
    The function plots the actual training data along with the regression line and evaluates the model
    if specified.

    :param train_df: The `train_df` parameter is a pandas DataFrame that contains the training data. It
    should have columns for the independent variable (`X_column_name`) and the dependent variable
    (`y_column_name`)
    :type train_df: pd.DataFrame
    :param X_column_name: The `X_column_name` parameter is the name of the column in the `train_df`
    DataFrame that contains the independent variable(s) or features for the regression model
    :type X_column_name: str
    :param y_column_name: The `y_column_name` parameter is the name of the column in the `train_df`
    DataFrame that contains the target variable or the variable you want to predict
    :type y_column_name: str
    :param regression_type: The `regression_type` parameter specifies the type of regression to be used.
    The default value is "lr", which stands for linear regression, defaults to lr
    :type regression_type: str (optional)
    :param eval_model: The `eval_model` parameter is a boolean flag that determines whether to evaluate
    the performance of the regression model. If `eval_model` is set to `True`, the `evaluate_model`
    function will be called to calculate and display evaluation metrics for the model's predictions. If
    `eval_model` is, defaults to False
    :type eval_model: bool (optional)
    """
    train_df = train_df.copy()

    X, y = X_y_split(train_df, X_column_name=X_column_name, y_column_name=y_column_name)
    X_df = pd.DataFrame(X, columns=[X_column_name])
    y_df = pd.DataFrame(y, columns=[y_column_name])

    # Line below is not actually getting any predictions,
    # just returning the regression line through the training data
    reggression_train_data = get_y_pred(X_df, y_df, X_df, regression_type)

    data = train_df[y_column_name]
    reg_train_df = pd.DataFrame(reggression_train_data, index=data.index)

    fig, ax = plt.subplots(figsize=(15, 5))
    data.plot(ax=ax, label="Actual", title="Actual vs Linear Regression Training Data")
    reg_train_df.plot(ax=ax, label="Regression Training Data")
    ax.legend(["Actual", "Predicted"])
    plt.show()

    if eval_model:
        evaluate_model(y_test=y, y_pred=reggression_train_data)


def combine_dfs(foldername: str, filename: str) -> pd.DataFrame:
    """
    The function `combine_dfs` combines multiple dataframes from files in a specified folder into a
    single dataframe.

    :param foldername: The `foldername` parameter is a string that represents the name of the folder
    where the CSV files are located
    :type foldername: str
    :param filename: The `filename` parameter is a string that represents the prefix of the files you
    want to combine. For example, if you have files named "file1.csv", "file2.csv", and "file3.csv", and
    you pass "file" as the `filename` parameter, the function will
    :type filename: str
    :return: a pandas DataFrame.
    """

    result_df = pd.DataFrame()
    for root, dirs, files in os.walk(foldername):
        for name in sorted(files):
            if name.startswith(filename):
                _filename = os.path.join(root, name)
                df = pd.read_csv(_filename, index_col=False)
                # df_reversed = df[::-1].reset_index(drop=True)
                df = df.sort_values(by="Posting Date").reset_index(drop=True)
                result_df = pd.concat([df, result_df])
    return result_df


def get_first_test_date(df: pd.DataFrame, X_column_name: str) -> datetime.datetime:
    """
    The function `get_first_test_date` returns the first date in a DataFrame's index or a specified
    column.

    :param df: A pandas DataFrame containing the data
    :type df: pd.DataFrame
    :param X_column_name: The X_column_name parameter is a string that represents the name of the column
    in the DataFrame that contains the dates or timestamps
    :type X_column_name: str
    :return: the first test date from the given DataFrame.
    """
    return df.index[0] if df.index.name else df[X_column_name][0]


def get_first_date_ts(df: pd.DataFrame, X_column_name: str) -> float:
    """
    The function `get_first_date_ts` returns the timestamp of the first date in a DataFrame column.

    :param df: The parameter `df` is a pandas DataFrame that contains the data
    :type df: pd.DataFrame
    :param X_column_name: The X_column_name parameter is a string that represents the name of the column
    in the DataFrame that contains the dates
    :type X_column_name: str
    :return: a float value, which is the timestamp of the first date in the specified column of the
    given DataFrame.
    """
    return get_first_test_date(df, X_column_name).timestamp()
