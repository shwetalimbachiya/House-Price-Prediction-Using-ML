# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def plot_null_features(data, amount_of_nulls):
    features = []
    nullValues = []
    for i in data:
        if (data.isna().sum()[i]) > amount_of_nulls and i != 'SalePrice':
            features.append(i)
            nullValues.append(data.isna().sum()[i])
    y_pos = np.arange(len(features))
    plt.bar(y_pos, nullValues, align='center', alpha=0.5)
    plt.xticks(y_pos, features)
    plt.ylabel('NULL Values')
    plt.xlabel('Features')
    plt.title('Features with more than {} NULL values'.format(amount_of_nulls))
    plt.show()


def get_correlated_features(data):
    covarianceMatrix = data.corr()
    listOfFeatures = [i for i in covarianceMatrix]
    setOfDroppedFeatures = set()
    for i in range(len(listOfFeatures)):
        # Avoid repetitions
        for j in range(i + 1, len(listOfFeatures)):
            feature1 = listOfFeatures[i]
            feature2 = listOfFeatures[j]
            # If the correlation between the features is > 0.8
            if abs(covarianceMatrix[feature1][feature2]) > 0.8:
                # Add one of them to the set
                setOfDroppedFeatures.add(feature1)
    # Tried different values of threshold and 0.8 was the one that gave the best results
    return setOfDroppedFeatures


def get_non_correlated_features_with_output(data):
    return [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.045]


def plot_features(data, feature):
    plt.plot(data.values, feature, 'bo')
    plt.ylabel('SalePrice')
    plt.xlabel('LotArea - Lot size in square feet')
    plt.title('SalePrice in function of LotArea using Decision Tree')
    plt.show()


def main():
    # Loading train data
    train = pd.read_csv('train.csv')

    # Loading test data (Write test.csv directory)
    test = pd.read_csv('test.csv')

    # Making full dataset by combining test and train
    data = train.append(test, sort=False)

    # Setting max null values to 1000
    max_null_values = 1000
    plot_null_features(data, max_null_values)

    # Drop columns that contain more than max_null_values NULL values
    data = data.dropna(axis=1, how='any', thresh=max_null_values)
    data = data.fillna(data.mean())

    # Replace categorical data with one-hot encoded data
    features_df = pd.get_dummies(data)

    features_df = features_df.drop(get_correlated_features(features_df), axis=1)
    features_df = features_df.drop(get_non_correlated_features_with_output(features_df), axis=1)

    # Removing target feature(SalePrice)
    del features_df['SalePrice']

    # Creating X and y arrays
    X = features_df.values
    y = data['SalePrice'].values

    # Splitting dataset into train(80%) and test(20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Applying SVR
    model = DecisionTreeRegressor(
        criterion='mse',
        max_features=85,
        min_samples_leaf=7,
        min_samples_split=18
    )

    # Fitting model to linear regression
    model.fit(X_train, y_train)

    # Prediction
    y_preds = model.predict(X_test)

    # Submit prediction
    index = len(data) - len(X_test)
    test = data.iloc[index:]
    output = pd.DataFrame({'Id': test.Id, 'SalePrice': y_preds})

    # Visualizing model by plotting graph
    plot_features(test['LotArea'], y_preds)

    # Generating output csv file
    output.to_csv('DT_prediction.csv', index=False)

    # Finding error in prediction
    mae = mean_absolute_error(y_test, y_preds)
    print("Error in testing using Decision Tree: ", mae)


if __name__ == "__main__":
    main()
