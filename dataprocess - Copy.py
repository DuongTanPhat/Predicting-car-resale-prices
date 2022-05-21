from tkinter.constants import END
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.stats import t
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.tools import add_constant
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from sklearn.compose import make_column_transformer, make_column_selector
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

pd.set_option('display.max.columns', None)
import os





data = pd.read_csv("input/audi.csv")

# check null
print(data.info())
# check mo ta
print(data.describe(include='all'))
# thống kê dữ liệu liên tục
sns.pairplot(data=data)
plt.subplots_adjust(bottom=0.054)
plt.show()

# check số lượng data đáng nghi
engine_0 = data[data["engineSize"]==0]["engineSize"]
print(engine_0.count())
mileage_300000_plus = data[data["mileage"]>300000]["mileage"]
print(mileage_300000_plus.count())
year_2000_minus = data[data["year"]<2000]["year"]
print(year_2000_minus.count())
tax_0 = data[data["tax"]==0]["tax"]
print(tax_0.count())
mpg_100_plus = data[data["mpg"]>400]["mpg"]
print(mpg_100_plus.count())
price_100000_plus = data[data["price"]>100000]["price"]
print(price_100000_plus.count())




#xử lý dữ liệu đáng nghi
data["log price"] = np.log10(data["price"])
data = data.loc[data["engineSize"]!=0]
data = data.loc[data["year"]<2021]
data = data.loc[data["year"]>2000]
data = data.loc[data["mileage"]<300000]
data = data.loc[data["mpg"]<100]
data["age_of_car"] = 2020 - data["year"]
data = data.drop(columns = ["year"])
data = data.drop(columns = ["price"])





# check số lượng và phân bổ giá cả theo năm
fig, axes = plt.subplots(2, 1, figsize=[15, 8])
sns.countplot(x = data["age_of_car"],ax=axes[0])
sns.barplot(x = data["age_of_car"], y = data["log price"],ax=axes[1])
plt.show()



# check số lượng và phân bổ nhiên liệu, hộp số
def plot_violin_and_count(x, y, data, axes):
    sns.violinplot(x=x, y=y, data=data, ax=axes[0], scale="width")
    g = sns.countplot(x=x, data=data, ax=axes[1])
    return g
fig, axes = plt.subplots(2, 2, figsize=[15, 8])
plot_violin_and_count(y="log price", x="fuelType", data=data, axes=axes[:, 0])
plot_violin_and_count(y="log price", x="transmission", data=data, axes=axes[:, 1])
plt.show()




#Thống kê số lượng các biến phân loại
categorical_columns = ["model", "fuelType", "transmission"]
def print_categorical_counts(df, columns):
    for col in columns:
        print(df.groupby(col)[col].count().sort_values().to_frame(name="count"))
print_categorical_counts(data, categorical_columns)



#Thống kê sô lượng biến model
model_count = data.groupby("model")["model"].count()
model_count.name = "model count"
data_sorted_model_count = (
    data.merge(model_count, on="model")
    .sort_values("model count", ascending=False)
    .drop("model count", axis="columns")
)
fig, axes = plt.subplots(2, 1, figsize=[15, 8])
sns.violinplot(x="model", y="log price", data=data_sorted_model_count, ax=axes[0], scale="width")
g = sns.countplot(x="model", data=data_sorted_model_count, ax=axes[1])
plt.show()



#Bỏ các dữ liệu kiểu phân loại có < 20 mẫu
def drop_almost_empty_categories(df, col, nmin=20):
    category_count = df.groupby(col)[col].count()
    for category_name, count in category_count.iteritems():
        if count < nmin:
            print(f"Dropping {category_name} in {col}")
            df = df[df[col] != category_name]
    return df
for col in categorical_columns:
    data = drop_almost_empty_categories(data, col)
data[categorical_columns] = data[categorical_columns].astype("category")

sns.pairplot(data=data)
plt.subplots_adjust(bottom=0.054)
plt.show()

def every_column_name_but(df,dependent):
    features = [col for col in df.columns if col != dependent]
    return features

def split_dependent(df, dependent="log price"):
    features = every_column_name_but(df, dependent)
    return df[features], df[dependent]

def make_cat_ohe(drop="first"):
    """Make a one hot encoder that only acts on categorical columns"""
    cat_transformer_tuple = (
        OneHotEncoder(drop=drop),
        make_column_selector(dtype_include="category"),
    )
    # ohe = make_column_transformer(cat_transformer_tuple, remainder=StandardScaler())
    ohe = make_column_transformer(cat_transformer_tuple, remainder="passthrough")
    return ohe

def scores_mean_and_std(scores,mse,rmse):
    """Finds mean and standard deviations of scores from `cross_validate`,
    and puts them in a dataframe."""
    scores = pd.DataFrame(scores)[["test_score", "train_score"]]
    mean = scores.mean().add_prefix("mean_")
    std = scores.std().add_prefix("std_")
    mse = pd.Series(mse,index=["mse"])
    rmse = pd.Series(rmse,index=["rmse"])
    mean_std = pd.concat((mean, std,mse,rmse))
    return mean_std

feature_cols = [
    "mileage",
    "age_of_car",
    "model",
    "transmission",
    "engineSize",
    "mpg",
    "fuelType",
    "tax"
]

feature_cols2 = [
    "mileage",
    "age_of_car",
    "model",
    "transmission",
    "engineSize",
    "mpg",
    "fuelType"
]

def calc_prediction_delta(y, y_pred, alpha=0.90, print_ratio_captured=False):
    """Calculates the half width of the prediction interval, in which the
    the fraction of values that fall within this interval is expected to
    be `alpha`.
    If `print_ratio_captured` is true, the ratio of values actually in the
    prediction interval is printed. This should be close to `alpha`.
    """
    n = len(y)
    resid = y - y_pred
    mean_resid = np.mean(y - y_pred)
    sN2 = 1 / (n - 1) * sum((resid - mean_resid) ** 2)
    dy = t.ppf((1 + alpha) / 2, n - 1) * np.sqrt(sN2) * (1 + 1 / n)
    if print_ratio_captured:
        print(
            "Ratio of values inside prediction interval:"
            + " {:.2f}, mean residual: {:.2g}".format(
                np.mean(np.abs(resid + mean_resid) < dy), mean_resid
            )
        )
    return dy
def eval_price_with_pred_interval(X, linreg, dy):
    y_predict = linreg.predict(X)
    y_pred_w_interval = pd.DataFrame(
        {"y": y_predict, "y-dy": y_predict - dy, "y+dy": y_predict + dy}
    )
    price = np.power(10, y_pred_w_interval).rename(
        {"y": "price", "y-dy": "lower", "y+dy": "upper"}, axis="columns"
    )
    return price


all_dys = {}
all_scores = {}
linreg = make_cat_ohe()

#Lựa chọn tính năng
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
for i in range(1, len(feature_cols) + 1):
    cols = ["log price"] + feature_cols[:i]
    X, y = split_dependent(data_train[cols], dependent="log price")
    linreg.fit(X,y)
    data_expanded_std = linreg.transform(X)
    X_test, y_test = split_dependent(data_test[cols], dependent="log price")
    data_expanded_std_test = linreg.transform(X_test)
    lin = LinearRegression()
    lin.fit(data_expanded_std,y)
    y_pred = lin.predict(data_expanded_std_test)
    resid = 10**y_test - 10**y_pred
    k = np.mean(np.square(resid))
    l = mean_squared_error(10**y_test,10**y_pred,squared=False)
    # print(l)
    scores = cross_validate(LinearRegression(), data_expanded_std, y, return_train_score=True)
    scores = scores_mean_and_std(scores,k,np.sqrt(k))
    all_scores[cols[-1]] = scores
    #Kiểm tra chỉ số VIF và p-value
    if i == len(feature_cols2):
        vif = pd.DataFrame()
        dfdata = pd.DataFrame(data_expanded_std.toarray(),columns=linreg.get_feature_names())
        condata = add_constant(dfdata)
        vif = pd.Series([variance_inflation_factor(condata.values, i) 
               for i in range(condata.shape[1])], 
              index=condata.columns)
        print("============vif============")
        print(vif)
        regressor = sm.OLS(y.array, condata).fit()
        print("============summary============")
        print(regressor.summary())
        print("============score============")
        print(r2_score(y_test,y_pred))
all_scores = pd.DataFrame(all_scores).T
all_scores.index.name = "Last added feature"
print(all_scores)
all_dys = {}
all_scores = {}
linreg = make_cat_ohe()

#Loại bỏ tính năng tax
data = data.drop(columns = ["tax"])

#Lựa chọn tính năng cho random forest
# data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
# for i in range(1, len(feature_cols2) + 1):
#     cols = ["log price"] + feature_cols2[:i]
#     X, y = split_dependent(data_train[cols], dependent="log price")
#     linreg.fit(X,y)
#     data_expanded_std = linreg.transform(X)
#     X_test, y_test = split_dependent(data_test[cols], dependent="log price")
#     data_expanded_std_test = linreg.transform(X_test)
#     lin = RandomForestRegressor()
#     lin.fit(data_expanded_std,y)
#     y_pred = lin.predict(data_expanded_std_test)
#     resid = 10**y_test - 10**y_pred
#     k = np.mean(np.square(resid))
#     l = mean_squared_error(10**y_test,10**y_pred,squared=False)
#     # print(l)
#     scores = cross_validate(RandomForestRegressor(), data_expanded_std, y, return_train_score=True)
#     scores = scores_mean_and_std(scores,k,np.sqrt(k))
#     all_scores[cols[-1]] = scores

   
all_scores = pd.DataFrame(all_scores).T
all_scores.index.name = "Last added feature"
print(all_scores)

#Diễn giải mô hình
def prepare_linear_coeffs(features, linreg, std=None):
    coeffs = pd.DataFrame(
        {
            "observable": features,
            "coef": linreg.coef_,
            "10^coef": np.power(10, linreg.coef_),
        }
    )
    coeffs = coeffs.set_index("observable")
    if std is not None:
        coeffs["std"] = std
        coeffs["coef*std"] = coeffs["std"] * coeffs["coef"]
        coeffs = coeffs.sort_values("coef*std", key=np.abs, ascending=False)
    return coeffs
def rename_ohe_features(features, cat_cols):
    for i, cat_col in enumerate(cat_cols):
        features = [
            feature.replace(f"onehotencoder__x{i}", cat_col) for feature in features
        ]
    return features
X, y = split_dependent(data, dependent="log price")
linreg.fit(X,y)
data_expanded_std = linreg.transform(X)
lin = LinearRegression()
lin.fit(data_expanded_std,y)
cat_cols = data.columns[data.dtypes == "category"]
features = linreg.get_feature_names()
features = rename_ohe_features(features, cat_cols)
std = pd.get_dummies(data).std()
print(prepare_linear_coeffs(features, lin, std))



# #So sánh các model
# for i in range(1, len(feature_cols) + 1):
#     cols = ["log price"] + feature_cols[:i]
#     X, y = split_dependent(data_train[cols], dependent="log price")
#     linreg.fit(X,y)
#     data_expanded_std = linreg.transform(X)
#     X_test, y_test = split_dependent(data_test[cols], dependent="log price")
#     data_expanded_std_test = linreg.transform(X_test)
#     lin = Ridge(tol=1e-9,alpha=0.927)
#     lin2 = LinearRegression()
#     lin3 = Lasso(alpha=0.00003327)
#     lin4 = RandomForestRegressor()
#     lin5 = ElasticNet(alpha=0.00005197)
#     lin.fit(data_expanded_std,y)
#     lin2.fit(data_expanded_std,y)
#     lin3.fit(data_expanded_std,y)
#     lin4.fit(data_expanded_std,y)
#     lin5.fit(data_expanded_std,y)
#     y_pred = lin.predict(data_expanded_std_test)
#     y_pred2 = lin2.predict(data_expanded_std_test)
#     y_pred3 = lin3.predict(data_expanded_std_test)
#     y_pred4 = lin4.predict(data_expanded_std_test)
#     y_pred5 = lin5.predict(data_expanded_std_test)
#     resid = 10**y_test - 10**y_pred
#     resid2 = 10**y_test - 10**y_pred2
#     resid3 = 10**y_test - 10**y_pred3
#     resid4 = 10**y_test - 10**y_pred4
#     resid5 = 10**y_test - 10**y_pred5
#     new = np.mean(np.square(resid))
#     new2 = np.mean(np.square(resid2))
#     new3 = np.mean(np.square(resid3))
#     new4 = np.mean(np.square(resid4))
#     new5 = np.mean(np.square(resid5))
#     print(new)
#     print(new2)
#     print(new3)
#     print(new4)
#     print(new5)
#     print(r2_score(y_test,y_pred3))
#     print(r2_score(y_test,y_pred2))
#     print("===")

def pred_ints(model, X, percentile=95):
    err_down = []
    err_up = []
    for x in range(X.shape[0]):
        preds = []
        for pred in model.estimators_:
            preds.append(pred.predict(X[x])[0])
        err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
        err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
    return err_down, err_up

def plot_residuals(X, y, linreg,X_test,y_test):
    linreg.fit(X, y)
    data_expanded_std = linreg.transform(X)
    data_expanded_std_test = linreg.transform(X_test)
    lin = RandomForestRegressor()
    lin.fit(data_expanded_std,y)
    # err_down, err_up = pred_ints(lin, data_expanded_std_test, percentile=90)
    
    # truth = y_test
    # correct = 0.
    # for i, val in enumerate(truth):
    #     if err_down[i] <= val <= err_up[i]:
    #         correct += 1
    # print (correct/len(truth))

    dy = calc_prediction_delta(y, lin.predict(data_expanded_std), alpha=0.99)
    y_pred = lin.predict(data_expanded_std_test)
    truth = y_test
    correct = 0.
    i=0
    for i, val in enumerate(truth):
        if y_pred[i] -dy <= val <= y_pred[i] + dy:
            correct += 1
    print (correct/len(truth))

    y_pred = lin.predict(data_expanded_std_test)
    resid = 10**y_test - 10**y_pred
    print(np.mean(np.square(resid)))
    print(lin.score(data_expanded_std_test,y_test))
    g = sns.jointplot(x=10**y_pred, y=resid, kind="scatter", joint_kws=dict(alpha=0.2))
    g.plot_joint(sns.kdeplot, color="r")
    g.ax_joint.set_xlabel(y.name + " (fitted)")
    g.ax_joint.set_ylabel("Residuals")
    plt.show()

cols = ["log price"] + feature_cols2[:len(feature_cols2) + 1]
X, y = split_dependent(data_train[cols], dependent="log price")
X_test, y_test = split_dependent(data_test[cols], dependent="log price")
plot_residuals(X, y, linreg,X_test,y_test)

