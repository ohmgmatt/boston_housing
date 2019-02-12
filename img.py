import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

headers = ['CRIM', 'ZN', 'NDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv("dataset.csv", sep='\s+',skiprows = 1, header=None, names = headers)

for item in headers:
    ax = df[item].plot.hist()
    plt.title('Histograms for ' + item)
    fig = ax.get_figure()
    fig.savefig('00_{}.png'.format(item))
    plt.close(fig)
