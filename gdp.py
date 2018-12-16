import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('gdp.csv')

#divide into train and validation set
train = data[:int(0.9*(len(data)))]
valid = data[int(0.8*(len(data))):]

#preprocessing (since arima takes univariate series as input)
train.drop('YEAR',axis=1,inplace=True)
valid.drop('YEAR',axis=1,inplace=True)

#plotting the data
train['GDP'].plot()
valid['GDP'].plot()
print(valid.index)
#building the model
from pyramid.arima import auto_arima
model = auto_arima(train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(train)









forecast = model.predict(n_periods=len(valid))
print(forecast)
forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

#plot the predictions for validation set
plt.plot(train, label='Train',color='green')
#plt.plot(valid, label='Valid',color='black')
plt.plot(valid.index,forecast, label='Prediction',color='yellow')
plt.show()
