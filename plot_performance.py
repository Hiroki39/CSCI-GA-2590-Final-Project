import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = np.load('data/test_performance.npz', allow_pickle=True)

# load data into dataframe
df = pd.DataFrame(data['gptturbo_multiarith'], columns=[1, 2, 4, 8], index=[
    "CoT", "VarCoT", "PyCot", "SymPy"])
df2 = pd.DataFrame(data["textdavinci_multiarith"], columns=[1, 2, 4, 8], index=[
    "CoT", "VarCoT", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4, 8], var_name='Shots', value_name='Accuracy')
df2 = df2.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4, 8], var_name='Shots', value_name='Accuracy')

sns.set_style("whitegrid")

df["Shots"] = df["Shots"].astype(str)
df["Model"] = "gpt-3.5-turbo"
df2["Shots"] = df2["Shots"].astype(str)
df2["Model"] = "text-davinci-002"

df = pd.concat([df, df2])

sns.catplot(data=df, x='Shots', y='Accuracy', hue='Prompting Method', col='Model',
            kind='bar', alpha=0.6, height=2.5, aspect=1.8, hue_order=["CoT", "VarCoT", "SymPy", "PyCot"], palette="dark")
# g.set_xticks([1, 2, 3, 4])
# g.set_xticklabels(['1', '2', '4', '8'])
plt.ylim(40, 100)

plt.savefig('images/multiarith.png', bbox_inches='tight')

# load data into dataframe
df = pd.DataFrame(data['gptturbo_aquarat'], columns=[1, 2, 4], index=[
    "CoT", "SymPy"])
df2 = pd.DataFrame(data["textdavinci_aquarat"], columns=[1, 2, 4], index=[
    "CoT", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4], var_name='Shots', value_name='Accuracy')
df2 = df2.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4], var_name='Shots', value_name='Accuracy')

df["Shots"] = df["Shots"].astype(str)
df["Model"] = "gpt-3.5-turbo"
df2["Shots"] = df2["Shots"].astype(str)
df2["Model"] = "text-davinci-002"

df = pd.concat([df, df2])

sns.catplot(data=df, x='Shots', y='Accuracy', hue='Prompting Method', col='Model',
            kind='bar', alpha=0.6, height=2.5, aspect=1.8, hue_order=["CoT", "SymPy"], palette="dark")

plt.savefig('images/aquarat.png', bbox_inches='tight')


df = pd.DataFrame(data['gptturbo_svamp'], columns=[1, 2, 4, 8], index=[
    "CoT", "VarCoT", "PyCot", "SymPy"])

# melt dataframe
df = df.reset_index(names=['Prompting Method']).melt(id_vars=['Prompting Method'], value_vars=[
    1, 2, 4, 8], var_name='Shots', value_name='Accuracy')

df["Shots"] = df["Shots"].astype(str)
df["Model"] = "gpt-3.5-turbo"

sns.catplot(data=df, x='Shots', y='Accuracy', hue='Prompting Method', col='Model',
            kind='bar', alpha=0.6, height=3, aspect=1, hue_order=["CoT", "VarCoT", "SymPy", "PyCot"], palette="dark")

plt.ylim(60, 90)

plt.savefig('images/svamp.png', bbox_inches='tight')
