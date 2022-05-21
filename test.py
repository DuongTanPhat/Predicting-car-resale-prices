from tkinter.constants import END
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain, combinations
import seaborn as sns
from scipy.stats import t
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet 
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
from tkinter.ttk import Button,  Entry, Combobox
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo,showwarning
from pickle import dump,load
warnings.filterwarnings('ignore')

pd.set_option('display.max.columns', None)
import os





window = tk.Tk()
window.title("Dự báo giá oto bán lại")
window.geometry('500x300')
tabControl = ttk.Notebook(window)
  
tabTrain = ttk.Frame(tabControl)
tabPredict = ttk.Frame(tabControl)
  
tabControl.add(tabPredict, text ='Predict')
tabControl.add(tabTrain, text ='Train')

tabControl.pack(expand = 1, fill ="both")

#tab predict
def find_longest_element(keys):
    return keys[np.argmax(list(map(len, keys)))]
def clicked_predict():
    global car,regressor
    features = find_longest_element(list(regressor.keys()))
    print(features)
    chosen_features = tuple(feature for feature in features if (feature == 'model' and carModelCb.get()!='') or 
    (feature == 'transmission' and transmissionCb.get()!='') or 
    (feature == 'mileage' and mileage.get()!='') or 
    (feature == 'fuelType' and fuelCb.get()!='') or 
    # ('tax' in feature and tax.get()!='') or
    (feature == 'mpg' and mpg.get()!='') or 
    (feature == 'engineSize' and engineSize.get()!='') or 
    (feature == 'age_of_car' and year.get()!='') 
    )
    if len(chosen_features) == 0:
        showwarning(title='Error', message="Vui lòng nhập ít nhất một thông tin có nghĩa!!!")
        return
    model_holder = regressor[chosen_features]
    model = model_holder["model"]
    # X = model_holder["X"]
    # Y = model_holder["Y"]
    dy = model_holder["dy"]
    score = model_holder["score"]
    ohe = model_holder["ohe"]
    arrayTest = []
    try:
        if(carModelCb.get()!='') and str(carModelCb['state'])!='disable':
            arrayTest.append(carModelCb.get())
        if transmissionCb.get()!='':
            arrayTest.append(transmissionCb.get())
        if mileage.get()!='':
            try:
                if int(mileage.get())<=0:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập số dặm hợp lệ (lớn hơn 0)")
                return
            arrayTest.append(int(mileage.get()))
        if fuelCb.get()!='':
            arrayTest.append(fuelCb.get())
        # if tax.get()!='':
        #     arrayTest.append(int(tax.get()))
        if mpg.get()!='':
            try:
                if float(mpg.get())<=0:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập giá trị mpg hợp lệ (lớn hơn 0)")
                return
            arrayTest.append(float(mpg.get()))
        if engineSize.get()!='':
            try:
                if float(engineSize.get())<=0:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập kích cỡ động cơ hợp lệ (lớn hơn 0)")
                return
            try:
                if float(engineSize.get())>=50:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập kích cỡ động cơ hợp lệ (đơn vị liter)")
                return
            arrayTest.append(float(engineSize.get()))
        if year.get()!='':
            try:
                if int(year.get())<=2000:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập năm sản xuất xe gần đây (lớn hơn 2000)")
                return
            try:
                if int(preYear.get())<2022:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập năm dự đoán đúng nghĩa (lớn hơn 2021)")
                return
            try:
                if int(year.get())>2021:
                    raise
            except:
                showwarning(title='Error', message="Vui lòng nhập năm sản xuất hợp lệ (nhỏ hơn 2022)")
                return
            arrayTest.append(int(preYear.get())-int(year.get()))
    except:
        showwarning(title='Error', message="Vui lòng nhập thông tin đúng định dạng")
        return
    
    aryy = pd.DataFrame(np.array(arrayTest).reshape(1,len(chosen_features)), columns = chosen_features)
    print(aryy)
    # col_dum = list(feature for feature in chosen_features if (feature == 'model') or (feature == 'transmission') or (feature == 'fuelType'))
    r = ohe.transform(aryy)
    a = model.predict(r)
    lower_90 = np.power(10, a[0] - dy)
    upper_90 = np.power(10, a[0] + dy)
    showinfo(title='Result', message="Giá dự đoán (£): "+str(round(10**a[0],2))+"\n\nKhoảng dự đoán 80%: \nTừ: "+str(round(lower_90,2))+"\nĐến: "+str(round(upper_90,2)))
    print(str(10**a[0]))
    # print(str(pow(10,a[0])))

labelChooseBrand = ttk.Label(tabPredict,text="Chọn hãng:")
labelChooseBrand.pack(fill='x', padx=5, pady=5)
stringChooseBrand = tk.StringVar()
brandCb = Combobox(tabPredict, width = 27, textvariable = stringChooseBrand)
brandCb['values'] = ('Audi', 
                          'BMW',
                          'Ford',
                          'Hyundi',
                          'Merc',
                          'Skoda',
                          'Toyota',
                          'Vauxhall',
                          'Vw')
brandCb.set('Audi')
brandCb.place(x=100,y=5)
brandCb['state'] = 'readonly'
brandCb.current()

labelCarModel = tk.Label(tabPredict,text="Model: ")
labelCarModel.place(x=5,y=30)
stringCarModel = tk.StringVar()
carModelCb = Combobox(tabPredict, width = 27, textvariable = stringCarModel)
carModelCb.place(x=100,y=30)
carModelCb['state'] = 'readonly'
carModelCb.current()

labelYear = tk.Label(tabPredict,text="Year: ")
labelYear.place(x=5,y=55)
year = tk.StringVar()
yearTextbox = Entry(tabPredict,width = 15,textvariable = year)
yearTextbox.place(x=100,y=55)

labelPredictYear = tk.Label(tabPredict,text="Predict Year: ")
labelPredictYear.place(x=250,y=55)
preYear = tk.StringVar()
preYearTextbox = Entry(tabPredict,width = 15,textvariable = preYear)
preYearTextbox.place(x=350,y=55)
preYear.set('2022')

labelTransmission = tk.Label(tabPredict,text="Transmission: ")
labelTransmission.place(x=5,y=80)
stringTransmission = tk.StringVar()
transmissionCb = Combobox(tabPredict, width = 27, textvariable = stringTransmission)
transmissionCb['values'] = ('Automatic', 
                          'Manual',
                          'Semi-Auto')
transmissionCb.place(x=100,y=80)
transmissionCb['state'] = 'readonly'
transmissionCb.current()

labelMileage = tk.Label(tabPredict,text="Mileage: ")
labelMileage.place(x=5,y=105)
mileage = tk.StringVar()
mileageTextbox = Entry(tabPredict,width = 15,textvariable = mileage)
mileageTextbox.place(x=100,y=105)

labelEngineSize = tk.Label(tabPredict,text="Engine size: ")
labelEngineSize.place(x=5,y=130)
engineSize = tk.StringVar()
engineSizeTextbox = Entry(tabPredict,width = 15,textvariable = engineSize)
engineSizeTextbox.place(x=100,y=130)

labelMpg = tk.Label(tabPredict,text="Mpg: ")
labelMpg.place(x=5,y=155)
mpg = tk.StringVar()
mpgTextbox = Entry(tabPredict,width = 15,textvariable = mpg)
mpgTextbox.place(x=100,y=155)

labelFuelType = tk.Label(tabPredict,text="Fuel type: ")
labelFuelType.place(x=5,y=180)
stringFuel = tk.StringVar()
fuelCb = Combobox(tabPredict, width = 27, textvariable = stringFuel)
fuelCb.place(x=100,y=180)
fuelCb['state'] = 'readonly'
fuelCb.current()

buttonPredict = Button(tabPredict,text="Predict", command=clicked_predict)
buttonPredict.place(x=5,y=230)


# Tab Train
def clicked_select_file_open():
    labelUrlFile.config(text="Đang chọn ... ")
    tabTrain.update()
    global url,url2,data,name
    global btn3, modelchoosen,featureschoosen,k_textbox
    url = tk.filedialog.askopenfilename(title="Select File", filetype=(("Data files","*.csv"),))
    if(url!=''):
        labelUrlFile.config(text=url)
        url2= url
        name= url.split("/")[-1].split(".")[0]
        try:
            data = pd.read_csv(url)
        except:
            buttonStatistic['state'] = 'disable'
            buttonTrain['state'] = 'disable'
            modelCb['state'] = 'disable'
            showwarning(title='Error', message="Read file failed! Please try again!")
            return
        buttonStatistic['state'] = 'normal'
        buttonTrain['state'] = 'normal'
        modelCb['state'] = 'readonly'
        # featureschoosen['state'] = 'readonly'
        # k.set('all')
        tabTrain.update()
    elif(url2!=''):
        labelUrlFile.config(text=url2)
        tabTrain.update()
    else:
        labelUrlFile.config(text="Chưa chọn")
        tabTrain.update()
def clicked_statistic_open():
    global data
    sns.pairplot(data=data)
    plt.subplots_adjust(bottom=0.054)
    plt.show()
def drop_almost_empty_categories(df, col, nmin=20):
    category_count = df.groupby(col)[col].count()
    for category_name, count in category_count.iteritems():
        if count < nmin:
            print(f"Dropping {category_name} in {col}")
            df = df[df[col] != category_name]
    return df
def powerset(iterable, start=0):
    #powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(start, len(s) + 1))
def make_cat_ohe(drop="first"):
    """Make a one hot encoder that only acts on categorical columns"""
    cat_transformer_tuple = (
        OneHotEncoder(drop=drop),
        make_column_selector(dtype_include="category"),
    )
    # ohe = make_column_transformer(cat_transformer_tuple, remainder="passthrough")
    ohe = make_column_transformer(cat_transformer_tuple, remainder=StandardScaler())
    return ohe
def split_dependent(df,data, dependent="log price"):
    features = every_column_name_but(df, dependent)
    return df[features], data[dependent]
def every_column_name_but(df, dependent):
    features = [col for col in df.columns if col != dependent]
    return features
def regression_model(model,X_train_transformed,y_train):
    """
    Will fit the regression model passed and will return the regressor object and the score
    """
    regressor = model
    regressor.fit(X_train_transformed, y_train)
    score = regressor.score(X_train_transformed, y_train)
    return regressor, score
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

def train_set_of_models(model,data3):
    models = {}
    features = data3.drop(columns = ["log price"]).columns
    i=0
    set_features =  pow(2,len(features))-1
    progress = 100/set_features
    for feature_set in powerset(features, start=1):
        regressor = LinearRegression()

        if (model=="LinearRegression"):
            regressor = LinearRegression()
        elif (model=="Ridge"):
            regressor = Ridge(tol=1e-9,alpha=0.927)
        elif (model=="Lasso"):
            regressor = Lasso(alpha=0.00003327)
        elif (model=="ElasticNet"):
            regressor = ElasticNet(alpha=0.00005197)
        elif (modelCb.get()=="RandomForestRegressor"):
            regressor = RandomForestRegressor()

        X = data3[list(feature_set)]
        ohe = make_cat_ohe()
        X_train, y_train = split_dependent(X,data3, dependent="log price")
        ohe.fit(X_train,y_train)
        X_transform = ohe.transform(X_train)
        regressor.fit(X_transform, y_train)
        score = regressor.score(X_transform, y_train)
        if (modelCb.get()=="RandomForestRegressor"):
            dy = calc_prediction_delta(y_train, regressor.predict(X_transform), alpha=0.99)
        else:
            dy = calc_prediction_delta(y_train, regressor.predict(X_transform), alpha=0.80)
        models[feature_set] = {"model": regressor,"score":score,"ohe":ohe,"dy":dy}
        i=i+progress
        labelTrainingProgress.config(text=str(i)+"%... ")
        tabTrain.update()
    return models
def clicked_train():
    global data,X_train, X_test, y_train, y_test,name
    buttonTrain['state'] = 'disable'
    labelTrainingProgress.config(text="0% Bắt đầu... ")
    tabTrain.update()
    try:
        categorical_columns = ["model",  "transmission", "fuelType"]
        data2 = data.copy()
        data2["log price"] = np.log10(data2["price"])
        data2 = data2.loc[data2["engineSize"]!=0]
        data2 = data2.loc[data2["year"]<2021]
        data2 = data2.loc[data["mpg"]>10]
        data2 = data2.loc[data["year"]>1990]
        if name=='audi':
            data2 = data2.loc[data["year"]>2000]
            data2 = data2.loc[data["mileage"]<300000]
            data2 = data2.loc[data["mpg"]<100]
        elif name=='bmw':
            data2 = data2.loc[data["price"]<100000]
            data2 = data2.loc[data["mpg"]<400]
        elif name=='ford':
            data2 = data2.loc[data["mpg"]<150]
        elif name=='hyundi':
            data2 = data2.loc[data["price"]<80000]
            data2 = data2.loc[data["mpg"]<150]
        elif name=='merc':
            data2 = data2.loc[data["price"]<80000]
        elif name=='skoda':
            data2 = data2.loc[data["price"]<80000]
        elif name=='toyota':
            data2 = data2.loc[data["year"]>2000]
        data2["age_of_car"] = 2020 - data2["year"]
        data2 = data2.drop(columns = ["year"])
        data2 = data2.drop(columns = ["price"])
        try:
            data2 = data2.drop(columns = ["tax"])
        except:
            print("Dont have tax column")
    # data2 = data2.drop(columns = ["fuelType"])

        data3 = data2.copy()
        for col in categorical_columns:
            data3 = drop_almost_empty_categories(data3, col)
        data3[categorical_columns] = data3[categorical_columns].astype("category")
        labelTrainingProgress.config(text="0% Tiền xử lý... ")
        tabTrain.update()
        dump(data3, open('data_'+name+'.pkl', 'wb'))
        regressor = train_set_of_models(modelCb.get(),data3)
        dump(regressor, open('regressor_'+name+'.pkl', 'wb'))
    except:
        labelTrainingProgress.config(text="")
        buttonTrain['state'] = 'normal'
        tabTrain.update()
        showwarning(title='Error', message="Data processing failed! Please try again!")
        return
    # regressor, score = regression_model(model,X_train_transformed,y_train)
    # dump(regressor, open('regressor_'+name+'.pkl', 'wb'))
    # adj_r2 = (1 - (1 - score) * (len(y_train) - 1) / (len(y_train) -X_train.shape[1] - 1))
    labelTrainingProgress.config(text="")
    buttonTrain['state'] = 'normal'
    tabTrain.update()
    showinfo(title='Info', message="Complete")
    
    # print("squared "+str(score))
    # print("squared adj"+str(adj_r2))
labelChooseFile = tk.Label(tabTrain,text="Chọn file train: ")
labelChooseFile.place(x=0,y=5)
buttonSelectFile = Button(tabTrain,text="Select File", command=clicked_select_file_open)
buttonSelectFile.place(x=100,y=5)

labelUrlFile = tk.Label(tabTrain,text="")
labelUrlFile.place(x=0,y=30)

buttonStatistic = Button(tabTrain,text="Thống kê",state='disable', command=clicked_statistic_open)
buttonStatistic.place(x=5,y=50)

labelChooseModel = tk.Label(tabTrain,text="Chọn model: ")
labelChooseModel.place(x=0,y=110)
stringModel = tk.StringVar()
modelCb = Combobox(tabTrain, width = 27, textvariable = stringModel)
modelCb['values'] = ('LinearRegression', 
                          'Ridge',
                          'Lasso',
                          'ElasticNet',
                          'RandomForestRegressor')
modelCb.set('LinearRegression')
modelCb.place(x=100,y=110)
modelCb['state'] = 'disable'
modelCb.current()

labelTrainingProgress = tk.Label(tabTrain,text="")
labelTrainingProgress.place(x=0,y=150)

buttonTrain = Button(tabTrain,text="Train",state='disable', command=clicked_train)
buttonTrain.place(x=5,y=170)

url=''
url2 = ''
model = LinearRegression()
features = None
data = None
name=''
X_train, X_test, y_train, y_test = None, None, None, None


def model_changed(event):
    global model
    if (modelCb.get()=="LinearRegression"):
        model = LinearRegression()
    elif (modelCb.get()=="Ridge"):
        model = Ridge(tol=1e-9,alpha=0.927)
    elif (modelCb.get()=="Lasso"):
        model = Lasso(alpha=0.00003327)
    elif (modelCb.get()=="ElasticNet"):
        model = ElasticNet(alpha=0.00005197)
    elif (modelCb.get()=="RandomForestRegressor"):
        model = RandomForestRegressor()
def car_changed(event):
    global car,regressor
    car = brandCb.get().lower()
    try:
        data = load(open('data_'+car+'.pkl', 'rb'))
        regressor = load(open('regressor_'+car+'.pkl', 'rb'))
        modelList = list(data['model'].unique())
        modelList.append('')
        modelList.sort()
        carModelCb['values'] = modelList
        fuelCb['values'] = list(data['fuelType'].unique())+['']
        transmissionCb['values'] = list(data['transmission'].unique())+['']
        carModelCb.set('')
        buttonPredict['state'] = 'normal'
    except:
        buttonPredict['state'] = 'disable'

modelCb.bind('<<ComboboxSelected>>', model_changed)
brandCb.bind('<<ComboboxSelected>>', car_changed)
car = brandCb.get().lower()
try:
    data = load(open('data_'+car+'.pkl', 'rb'))
    regressor = load(open('regressor_'+car+'.pkl', 'rb'))
    modelList = list(data['model'].unique())
    modelList.append('')
    modelList.sort()
    carModelCb['values'] = modelList
    fuelCb['values'] = list(data['fuelType'].unique())+['']
    transmissionCb['values'] = list(data['transmission'].unique())+['']
    # modelcarchoosen.current(0)
    buttonPredict['state'] = 'normal'
except:
    buttonPredict['state'] = 'disable'
window.mainloop()    





















