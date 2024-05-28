import googlemaps
import pandas as pd

import geopy
import geopy.point

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler

# API key
API_KEY = 'AIzaSyD7YNeEWrMAy1E86S78K6BsmOGPfhRqbVQ'

# initialize the client
gmaps = googlemaps.Client(key=API_KEY)

cols = [
    'city',
    'location_lat',
    'location_lng',
    'numberOfGuests',
    'minNights',
    'roomTypeCategory',
    'bedroomLabel',
    'bedLabel',
    'bathroomLabel',
    'pricing',
    'stars'
]

data = pd.read_csv('airbnb_10k.csv', usecols=cols)

print(data.head())

# num of missing

for col in data.columns:
    print(col + ', Number of Missing Values:', len(data[col][data[col].isnull()]))

# remove NaN values

original_len = len(data)
print(original_len)
data = data.dropna(how='any', subset=['location_lat', 'location_lng',
                                      'bedroomLabel', 'bedLabel', 'bathroomLabel', 'stars'])
print('\nRemoved NaN Value Count:', original_len - len(data))

print("Remaining: ", len(data))

# formatting price

data['pricing'] = (data['pricing'].replace(r'[^-+\d.]', '').astype(float))

# remove words from bedroomLabel, bedLabel, bathroomLabel column

data['bedroomLabel'] = data['bedroomLabel'].str.extract('(\\d+)', expand=False).astype(float)
data['bedLabel'] = data['bedLabel'].str.extract('(\\d+)', expand=False).astype(float)
data['bathroomLabel'] = data['bathroomLabel'].str.extract('(\\d+)', expand=False).astype(float)

print(data.tail())

# drop işe yaramaz values
print('Number of Guests 0:', len(data[data['numberOfGuests'] == 0]))
print('Number of Bedrooms 0:', len(data[data['bedroomLabel'] == 0]))
print('Number of Beds 0:', len(data[data['bedLabel'] == 0]))
print('Number of Price 0.00:', len(data[data['pricing'] == 0.00]))

# roomTypeCategory one-hot encoding
print(data['roomTypeCategory'].unique())

data = pd.get_dummies(data, columns=['roomTypeCategory'], prefix='roomType')

data['roomType_entire_home'] = data['roomType_entire_home'].astype(int)
data['roomType_hotel_room'] = data['roomType_hotel_room'].astype(int)
data['roomType_private_room'] = data['roomType_private_room'].astype(int)

data['stars'] = data['stars'].str.replace('"', '')
data['stars'] = data['stars'].str.replace(',', '.')
data['stars'] = data['stars'].astype(float)


print(data.head())

print("Remaining: ", len(data))

district_zipcodes = {
    "Adalar": "34975",
    "Arnavutköy": "34275",
    "Ataşehir": "34758",
    "Avcılar": "34315",
    "Bağcılar": "34200",
    "Bahçelievler": "34180",
    "Bakırköy": "34158",
    "Başakşehir": "34494",
    "Bayrampaşa": "34035",
    "Beşiktaş": "34353",
    "Beykoz": "34810",
    "Beylikdüzü": "34520",
    "Beyoğlu": "34421",
    "Büyükçekmece": "34500",
    "Çatalca": "34550",
    "Çekmeköy": "34794",
    "Esenler": "34230",
    "Esenyurt": "34517",
    "Eyüpsultan": "34060",
    "Fatih": "34096",
    "Gaziosmanpaşa": "34245",
    "Güngören": "34160",
    "Kadıköy": "34744",
    "Kağıthane": "34403",
    "Kartal": "34862",
    "Küçükçekmece": "34307",
    "Maltepe": "34854",
    "Pendik": "34893",
    "Sancaktepe": "34885",
    "Sarıyer": "34396",
    "Silivri": "34582",
    "Sultanbeyli": "34920",
    "Sultangazi": "34265",
    "Şile": "34990",
    "Şişli": "34360",
    "Tuzla": "34959",
    "Ümraniye": "34773",
    "Üsküdar": "34660",
    "Zeytinburnu": "34025"
}

district_index = {
    "Adalar": "68.133",
    "Arnavutköy": "27.857",
    "Ataşehir": "48.888",
    "Avcılar": "31.818",
    "Bağcılar": "35.313",
    "Bahçelievler": "31.666",
    "Bakırköy": "76.219",
    "Başakşehir": "43.043",
    "Bayrampaşa": "35.536",
    "Beşiktaş": "88.235",
    "Beykoz": "72.222",
    "Beylikdüzü": "29.950",
    "Beyoğlu": "60.000",
    "Büyükçekmece": "36.250",
    "Çatalca": "32.777",
    "Çekmeköy": "36.666",
    "Esenler": "33.871",
    "Esenyurt": "21.250",
    "Eyüpsultan": " 45.000",
    "Fatih": "32.000",
    "Gaziosmanpaşa": "32.407",
    "Güngören": "36.666",
    "Kadıköy": "88.888",
    "Kağıthane": "41.428",
    "Kartal": "43.030",
    "Küçükçekmece": "34.347",
    "Maltepe": "48.076",
    "Pendik": "35.386",
    "Sancaktepe": "29.500",
    "Sarıyer": "103.881",
    "Silivri": "29.062",
    "Sultanbeyli": "28.333",
    "Sultangazi": "27.272",
    "Şile": "54.677",
    "Şişli": "54.677",
    "Tuzla": "34959",
    "Ümraniye": "40.888",
    "Üsküdar": "61.428 ",
    "Zeytinburnu": "47.222"
}


# convert lat/lng to zipcode

import geopy
import pandas as pd


def get_zipcode(row, gmaps):
    try:
        result = gmaps.reverse_geocode((row['location_lat'], row['location_lng']))
        if result:
            for component in result[0]['address_components']:
                if 'postal_code' in component['types']:
                    return component['long_name']
    except Exception as e:
        print(f"Error: {e}")
    return None


data['location_lat'] = data['location_lat'].str.replace(',', '.').astype(float)
data['location_lng'] = data['location_lng'].str.replace(',', '.').astype(float)
data['zipcode'] = data.apply(get_zipcode, axis=1, gmaps=gmaps)


data.to_csv('airbnb_clean_data.csv', index=False)

data = pd.read_csv('airbnb_clean_data.csv')
print(data.head())


def update_zipcode(row, district_zipcodes):
    city = row['city']
    if city in district_zipcodes:
        return district_zipcodes[city]
    return row['zipcode']

# Apply update_zipcode to update zipcodes to district level zipcodes
data['zipcode'] = data.apply(update_zipcode, axis=1, district_zipcodes=district_zipcodes)
data['district_index'] = data['city'].map(district_index)

data['roomType_shared_room'] = data['roomType_shared_room'].astype(int)
data.to_csv('airbnb_clean_data.csv', index=False)

print(data.head())

# CLEAN DATAYLA DEVAM EDİYORUZ

data = pd.read_csv('airbnb_clean_data.csv')
original_len = len(data)
data = data.dropna(how='any', subset=['city', 'location_lat', 'location_lng',
                                      'bedroomLabel', 'bedLabel', 'bathroomLabel', 'stars', 'district_index'])
print('\nRemoved NaN Value Count:', original_len - len(data))

# some visualizations according to the pricing col

import matplotlib.pyplot as plt
import seaborn as sns

# sns.distplot(
#     data['pricing'], norm_hist=False
# ).set(xlabel='pricing', ylabel='Count')
# data.hist(figsize=(15, 30), layout=(9, 3))

plt.figure(figsize=(10, 6))
sns.distplot(
    data['pricing'], norm_hist=False
).set(xlabel='Pricing', ylabel='Count', title='Distribution of Pricing')
plt.title('Distribution of Pricing')  # Başlık ekleme
plt.show()


data.hist(figsize=(15, 30), layout=(9, 3))
plt.suptitle('Histograms of Airbnb Features', y=1.02)  # Ana başlık ekleme
plt.tight_layout()
plt.show()


# for outlier detect
import numpy as np

for i in range(0, 100, 10):
    var = data["pricing"].values
    var = np.sort(var, axis=None)
    print("{} percentile value is {}".format(i, var[int(len(var) * (float(i) / 100))]))
print("100 percentile value is ", var[-1])
for i in range(90, 100, 1):
    var = data["pricing"].values
    var = np.sort(var, axis=None)
    print("{} percentile value is {}".format(i, var[int(len(var) * (float(i) / 100))]))
print("100 percentile value is ", var[-1])
outlier = data.pricing >= 20000
print(outlier.head())
outlier_dt = data[outlier]
print(outlier_dt["pricing"])

# ML algorithms before remove outliers

data = data.dropna(how='any', subset=['location_lat', 'location_lng',
                                      'bedroomLabel', 'bedLabel', 'bathroomLabel', 'stars', 'pricing'])

X = data[['numberOfGuests', 'minNights', 'roomType_entire_home',
          'roomType_hotel_room', 'roomType_private_room', 'roomType_shared_room', 'bedroomLabel',
          'bedLabel', 'bathroomLabel', 'stars', 'district_index']]
y = data['pricing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

results = {}
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['Linear Regression'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
model = SVR(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['SVR'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['Random Forest'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
results['Decision Tree'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
results['KNN'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}

# print results
for model, metrics in results.items():
    print(f"{model} - Mean Squared Error: {metrics['mse']:.4f}, R2 Score: {metrics['r2']:.4f}")

# visualize result
model_names = list(results.keys())
mse_values = [metrics['mse'] for metrics in results.values()]
r2_values = [metrics['r2'] for metrics in results.values()]
# mean squared erorr
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error of Different ML Models (before removing outliers)')
plt.show()
# r2 score
plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_values, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('R2 Score of Different ML Models (before removing outliers)')
plt.show()

# heatmap for corr
corr = X.corr(method='kendall')
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data['pricing'])
plt.xlabel('Pricing')
plt.title('Boxplot of Pricing')
plt.show()


# view outliers

Q1 = data['pricing'].quantile(0.25)
Q3 = data['pricing'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.0 * IQR
upper_bound = Q3 + 1.0 * IQR
outliers = data[data['pricing'] >= 20000]
print("Outliers:")
print(outliers['pricing'])

# color outliers

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data.index, y=data['pricing'], color='blue')
sns.scatterplot(x=outliers.index, y=outliers['pricing'], color='red')
plt.xlabel('Index')
plt.ylabel('Pricing')
plt.title('Pricing with Outliers Highlighted')
plt.show()


# remove outliers

# data_no_outliers = data[(data['pricing'] >= lower_bound) & (data['pricing'] <= upper_bound)]
data_no_outliers = data[data['pricing'] <= 20000]
print("Original data count:", data.shape)
print("Data count after removing outliers:", data_no_outliers.shape)


# boxplot without outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data_no_outliers['pricing'])
plt.xlabel('Pricing')
plt.title('Boxplot of Pricing After Removing Outliers')
plt.show()


# apply regression algorithms

X = data_no_outliers[['numberOfGuests', 'minNights', 'roomType_entire_home',
                      'roomType_hotel_room', 'roomType_private_room', 'roomType_shared_room','bedroomLabel',
                      'bedLabel', 'bathroomLabel', 'stars', 'district_index']]
y = data_no_outliers['pricing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

results = {}
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['Linear Regression'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
model = SVR(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['SVR'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
results['Random Forest'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
results['Decision Tree'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
results['KNN'] = {
    'mse': mean_squared_error(y_test, y_pred),
    'r2': r2_score(y_test, y_pred)
}
# print results
for model, metrics in results.items():
    print(f"{model} - Mean Squared Error: {metrics['mse']:.4f}, R2 Score: {metrics['r2']:.4f}")
# visualize result
model_names = list(results.keys())
mse_values = [metrics['mse'] for metrics in results.values()]
r2_values = [metrics['r2'] for metrics in results.values()]
# mean squared erorr
plt.figure(figsize=(10, 6))
plt.bar(model_names, mse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error of Different ML Models')
plt.show()
# r2 score
plt.figure(figsize=(10, 6))
plt.bar(model_names, r2_values, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('R2 Score of Different ML Models')
plt.show()
