import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = np.load('data/test_performance.npz', allow_pickle=True)

# load data into dataframe
df = pd.DataFrame(data['gptturbo_multiarith'], columns=[1, 2, 4, 8], index=[
    "CoT", "VarCoT", "PyCot", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4, 8], var_name='Shots', value_name='Accuracy')

sns.set_style("whitegrid")

df["Shots"] = df["Shots"].astype(str)

g = sns.catplot(data=df, x='Shots', y='Accuracy',
                hue='Prompting Method', kind='bar', alpha=0.6, height=6, aspect=1.5)
# g.set_xticks([1, 2, 3, 4])
# g.set_xticklabels(['1', '2', '4', '8'])
plt.ylim(90, 100)
plt.title('GPT-Turbo Multi-Arithmetic Dataset Accuracy')

plt.savefig('images/gptturbo_multiarith.png', bbox_inches='tight')

df = pd.DataFrame(data['textdavinci_multiarith'], columns=[2, 4, 8], index=[
    "CoT", "VarCoT", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    2, 4, 8], var_name='Shots', value_name='Accuracy')

df["Shots"] = df["Shots"].astype(str)

g = sns.catplot(data=df, x='Shots', y='Accuracy',
                hue='Prompting Method', kind='bar', alpha=0.6, height=6, aspect=1.5)

plt.ylim(50, 100)
plt.title('TextDavinci Multi-Arithmetic Dataset Accuracy')

plt.savefig('images/textdavinci_multiarith.png', bbox_inches='tight')

df = pd.DataFrame(data['textdavinci_aquarat'], columns=[1, 2, 4], index=[
    "CoT", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4], var_name='Shots', value_name='Accuracy')

df["Shots"] = df["Shots"].astype(str)

g = sns.catplot(data=df, x='Shots', y='Accuracy',
                hue='Prompting Method', kind='bar', alpha=0.6, height=6, aspect=1.5)

plt.ylim(0, 40)
plt.title('TextDavinci AQUA-RAT Dataset Accuracy')

plt.savefig('images/textdavinci_aquarat.png', bbox_inches='tight')
