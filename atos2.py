import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import csv
from scipy.io import loadmat
#%matplotlib inline

data = pd.read_csv('mydata.csv')

F1 = data[:1]
F2 = data[:2]

print(F1.shape)
print(F2.shape)

fig, ax, plt.subplots(figsize=(20, 20))
ax.scatter(X[:,0], X[:,1])
