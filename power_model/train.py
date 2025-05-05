from __future__ import print_function
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def load_data(path):
    df = pd.read_csv(path)
    df = df[df['Power'] < 1700]
    X = df[['Bitrate','FileSize','Quality','Motion']].values
    pixel_rate = (df['Height'] * df['Width']).values.reshape(-1,1)
    return np.hstack([X, pixel_rate]), df['Power'].values

# 1) load & split
X, y = load_data('dataset_all_videos_galaxy.csv')
# optionally shuffle / train-test split here if you like

# 2) build model
model = Sequential()
model.add(Dense(7,  input_dim=X.shape[1],
                kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1,  kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

# 3) train
print('Training...')
model.fit(X, y,
          epochs=100, batch_size=10, verbose=2)

# 4) save
model.save('power_model_py2.h5')
print('Saved model to power_model_py2.h5')

