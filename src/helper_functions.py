"""
This script contains functions that will be used in different scripts
"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

def plot_age_distribution(df, save_path):
    plt.figure()
    min_age = df['age'].min()
    max_age = df['age'].max()
    plt.hist(df['age'], bins=int(max_age-min_age), range=(min_age, max_age))
    plt.xlabel('Age')
    plt.ylabel('# of Subjects')
    plt.savefig(save_path)
    plt.close()
