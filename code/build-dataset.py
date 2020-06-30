import pandas as pd
import numpy as np

#Get Flags and Execution Times
flags_csv = pd.read_csv('../raw_data/cBench_onPandaboard_24app_5ds.csv')


y = flags_csv.iloc[:,8:-1]
y.insert(0, 'APP_NAME', list(flags_csv['APP_NAME']), True,)
print('y:', y.shape)
flags = flags_csv.iloc[:,:8]
flags.insert(flags.shape[1], 'code_size', flags_csv['code_size'] ,True)
print('flags:', flags.shape)

flags = flags.replace(to_replace='X', value=0)
flags = flags.replace(to_replace='-', value=1, regex=True)
flags.head(n=5)


#Get Static
static_csv = pd.read_csv('../raw_data/ft_Milepost_cbench.csv')
print('static:', static_csv.shape)

static = static_csv.drop(['DATASET'], axis=1)
static.drop_duplicates(keep='first', inplace=True)
static = static.reset_index()
static.head()


#Get Dynamic
dynamic_csv = pd.read_csv('../raw_data/ft_MICA_cbench.csv')
print('dynamic:', dynamic_csv.shape)

dynamic = dynamic_csv.rename(columns={'APPLICATION_NAME': 'APP_NAME'})
dynamic_list = [pd.DataFrame(dynamic[dynamic['DATASET'] == val].iloc[:24]).drop(['DATASET'], axis=1).rename(columns=lambda x: x + '_' + val[-1:]).reset_index() for val in list(dynamic['DATASET'].unique())]
dynamic = pd.concat(dynamic_list, axis=1, sort=False).drop('index', axis=1)
dynamic = dynamic.drop(['APP_NAME_' + str(i) for i in range(2,6)], axis=1).rename(columns={'APP_NAME_1': 'APP_NAME'})
dynamic.head()


apps = list(static['APP_NAME'].unique())

flags_temp = flags[flags['APP_NAME'] == apps[0]]

static_temp = pd.DataFrame(static[static['APP_NAME'] == apps[0]])
static_temp = pd.concat([static_temp]*128, ignore_index=True).drop('APP_NAME', axis=1)

dynamic_temp = pd.DataFrame(dynamic[dynamic['APP_NAME'] == apps[0]])
dynamic_temp = pd.concat([dynamic_temp]*128, ignore_index=True).drop('APP_NAME', axis=1)

y_temp = y[y['APP_NAME'] == apps[0]].reset_index(drop=True).drop('APP_NAME', axis=1)

data = pd.concat([flags_temp, static_temp, dynamic_temp, y_temp], axis=1, sort=False)

for app in apps[1:]:
    flags_temp = pd.DataFrame(flags[flags['APP_NAME'] == app]).reset_index(drop=True)
    
    static_temp = pd.DataFrame(static[static['APP_NAME'] == app])
    static_temp = pd.concat([static_temp]*128, ignore_index=True).reset_index(drop=True).drop('APP_NAME', axis=1)
    
    dynamic_temp = pd.DataFrame(dynamic[dynamic['APP_NAME'] == app])
    dynamic_temp = pd.concat([dynamic_temp]*128, ignore_index=True).reset_index(drop=True).drop('APP_NAME', axis=1)
    
    y_temp = y[y['APP_NAME'] == app].reset_index(drop=True).drop('APP_NAME', axis=1)
    
    data_app = pd.concat([flags_temp, static_temp, dynamic_temp, y_temp], axis=1, sort=False)
    
    data = pd.concat([data, data_app], axis=0, sort=False)

data = data.reset_index(drop=True).drop('index', axis=1)


data.head()


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(data.iloc[:,1:].values)
scaled_data = pd.concat([data['APP_NAME'], pd.DataFrame(scaler.transform(data.iloc[:,1:].values))], axis=1, ignore_index=True)
scaled_data.columns = data.columns


def data_to_csv(dataset, csv_name):
    if str(csv_name[-4:]) != '.csv':
        csv_name = str(csv_name) + '.csv'
    with open('../data/' + csv_name, 'w') as dataset_csv:
        dataset_csv.write(dataset.to_csv(index=False))


data_to_csv(data, 'data')


scaled_data.head()