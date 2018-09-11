import pickle
import numpy as np
from collections import Counter

val_data = pickle.load(open('validate_raw_data.p', 'rb'))

val_labels = val_data[1]
windspeed = np.array(val_labels)[:,1]

def cyclone_classification(wind_speed):
    # Wind speed is in knots
    if wind_speed >= 137:
        return 'h5'
    if wind_speed <= 136 and wind_speed >= 113:
        return 'h4'
    if wind_speed <= 112 and wind_speed >= 96:
        return 'h3'
    if wind_speed <= 95 and wind_speed >= 83:
        return 'h2'
    if wind_speed <= 82 and wind_speed >= 64:
        return 'h1'
    if wind_speed <= 63 and wind_speed >= 34:
        return 'ts'
    if wind_speed <= 33 and wind_speed > 0:
        return 'td'

    return None

cat = [cyclone_classification(item) for item in windspeed]
print(Counter(cat))

coordinates = [item[0] for item in val_labels]
# print(np.array(coordinates))
# with open('test_raw_data.p', 'wb') as f:
#     pickle.dump([test_data,test_labels], f, protocol=pickle.HIGHEST_PROTOCOL)