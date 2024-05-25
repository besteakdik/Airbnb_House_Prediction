import pandas as pd

import geopy
import geopy.point

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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

data = pd.read_csv('airbnb_big_data.csv', usecols=cols)

print(data.head())

#num of missing

for col in data.columns:
    print(col + ', Number of Missing Values:', len(data[col][data[col].isnull()]))
    
# remove NaN values

original_len = len(data)
print(original_len)
data = data.dropna(how='any', subset=['city', 'location_lat', 'location_lng', 
                                      'bedroomLabel', 'bedLabel', 'bathroomLabel', 'stars'])
print('\nRemoved NaN Value Count:', original_len - len(data))

print("Remaining: ", len(data))

# formatting price

data['pricing'] = (data['pricing'].replace(r'[^-+\d.]', '').astype(float))


#remove words from bedroomLabel, bedLabel, bathroomLabel column

data['bedroomLabel'] = data['bedroomLabel'].str.extract('(\\d+)', expand=False).astype(float)
data['bedLabel'] = data['bedLabel'].str.extract('(\\d+)', expand=False).astype(float)
data['bathroomLabel'] = data['bathroomLabel'].str.extract('(\\d+)', expand=False).astype(float)

print(data.tail())

# drop işe yaramaz values
print('Number of Guests 0:', len(data[data['numberOfGuests'] == 0]))
print ('Number of Bedrooms 0:', len(data[data['bedroomLabel'] == 0]))
print ('Number of Beds 0:', len(data[data['bedLabel'] == 0]))
print ('Number of Price 0.00:', len(data[data['pricing'] == 0.00]))


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

# convert lat/lng to zipcode

import geopy
import pandas as pd

def get_zipcode(row, geolocator):
    location = geolocator.reverse((row['location_lat'], row['location_lng']))
    #print(location.raw)
    return location.raw['address']['postcode']

geolocator = geopy.Nominatim(user_agent='user_agent')

data['location_lat'] = data['location_lat'].str.replace(',', '.')
data['location_lng'] = data['location_lng'].str.replace(',', '.')

data['location_lat'] = data['location_lat'].astype(float)
data['location_lng'] = data['location_lng'].astype(float)


zipcodes = data.apply(get_zipcode, axis=1, geolocator=geolocator)

data['zipcode'] = zipcodes
#print(zipcodes)

data.to_csv('airbnb_clean_data.csv', index=False)

print(data.head())

data = pd.read_csv('airbnb_clean_data.csv')

for col in data.columns:
    print(col + ', Number of Missing Values:', len(data[col][data[col].isnull()]))
    


def regression_model(data):

    X = data[['numberOfGuests', 'minNights', 'roomType_entire_home', 
              'roomType_hotel_room', 'roomType_private_room', 'bedroomLabel', 
              'bedLabel', 'bathroomLabel', 'zipcode', 'stars']]
    y = data['pricing']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model performansını değerlendirme
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("LR Mean Squared Error:", mse)
    print("LR R^2 Score:", r2)

    return model

from sklearn.svm import SVR

X = data[['numberOfGuests', 'minNights', 'roomType_entire_home', 
          'roomType_hotel_room', 'roomType_private_room', 'bedroomLabel', 
          'bedLabel', 'bathroomLabel', 'stars']]
y = data['pricing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("SVR Mean Squared Error:", mse)
print("SVR R^2 Score:", r2)

from sklearn.ensemble import RandomForestRegressor

X = data[['numberOfGuests', 'minNights', 'roomType_entire_home', 'roomType_hotel_room', 
          'roomType_private_room', 'bedroomLabel', 'bedLabel', 'bathroomLabel', 'stars']]
y = data['pricing']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RF Mean Squared Error:", mse)
print("RF R^2 Score:", r2)

regression_model = regression_model(data)

print(data.head())
