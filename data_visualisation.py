import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('dataset/extracted_data.csv')
replace_rule = {'weather_condition' : {'rain': 'Rain', 'sunshine' : 'Sunshine'}}
dataset = dataset.replace(replace_rule)
dataset = dataset.rename(columns={'weather_condition' : 'Weather Condition'})

sns.relplot(data=dataset, x='wind_direction', y='pressure', hue='Weather Condition')
# sns.heatmap(dataset.corr())
plt.xlabel('Wind Direction')
plt.ylabel('Pressure')
plt.title('Wind Direction vs Pressure')

plt.show()