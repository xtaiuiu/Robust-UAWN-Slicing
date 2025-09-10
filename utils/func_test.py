import pandas as pd
from matplotlib import pyplot as plt

data = {'Year': [2010, 2011, 2012, 2013, 2014],
        'Sales': [200, 220, 250, 270, 300]}
df = pd.DataFrame(data)
df.plot(x='Year', y='Sales', kind='pie')
df.plot(x='Year', y='Sales', kind='bar')
df.plot(x='Year', y='Sales', kind='scatter')
df.plot(x='Year', y='Sales', kind='hist')
plt.show()