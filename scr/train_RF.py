import pandas as pd
import time as tm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

file_name = './export_all_poi.csv'
trees=100
train = 'raw' 
#train = 'context'

if __name__ == "__main__":
    file = pd.read_csv(file_name) 

    if train == 'raw':
        data=file[[ 'month','day','doy','week','hour']] 
    else:
        data=file[['month','day','doy','dow','week','hour','holiday','festive','temperature','humidity','pressure','wind','rain']]

    target = file[[ 'presenze']]

    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2, random_state=0)

    n = MinMaxScaler()
    X_train = n.fit_transform(X_train)
    X_test = n.transform(X_test)

    begin_time = tm.time()  #start time training

    forest_reg = RandomForestRegressor(n_estimators=trees, random_state=0)
    forest_reg.fit(X_train, Y_train.values.ravel())

    end_time = tm.time()    #end time training


    data_predictions = forest_reg.predict(X_test)

    mape = round(mean_absolute_percentage_error(Y_test, data_predictions), 5)
    tot_time = round(end_time - begin_time, 2)

    title = '\n---------------result--------------------\nFile name: {}\nData tipe: {}\n'.format(file_name, train)
    output = 'n_estimator {}: \t time: {}sec \t mape: {}\n'.format(trees, tot_time, mape)

    print(title, output)
