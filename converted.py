# -*- coding: utf-8 -*-
"""vitals.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16HzRwsB9qtlYSa4qgVEP567fTjYdTwqK
"""

import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())

"""
Time,PR_Val,Temp_Val,Sys_Val,Resp_Val,SPO2
"""

def assign_mews_score_to_pulse_rate(pulse_rate):
    # Define MEWS score ranges for Pulse Rate
    if 51<= pulse_rate <= 100:
        return 0  # Normal
    elif 101 < pulse_rate <= 110:
        return 1  # Mildly elevated
    elif 41 < pulse_rate <= 50:
        return 1  # Mildly elevated
    elif 111 < pulse_rate <= 129:
        return 2  # Elevated
    elif pulse_rate <= 40:
        return 2  # Elevated
    elif pulse_rate >= 130:
        return 3  # High
    else:
        return 3  # Very high

def assign_mews_score_to_spo2(spo2):
    # Define MEWS score ranges for Spo2
    if 95 <= spo2 :
        return 0  # Normal
    elif 90 <= spo2 <=94:
        return 1  # Mildly reduced
    elif 86 <= spo2 <= 89:
        return 2  # Moderately reduced
    elif  spo2 <= 85:
        return 3  # Severely reduced
    else:
        return 3  # Very severely reduced

def assign_mews_score_to_respiratory_rate(respiratory_rate):
    # Define MEWS score ranges for Respiratory Rate
    if 9 <= respiratory_rate <= 14:
        return 0  # Normal
    elif 15 <= respiratory_rate <= 20:
        return 1  # Mildly elevated
    elif 21 <= respiratory_rate <= 29:
        return 2  # Elevated
    elif 30 <= respiratory_rate :
        return 3  # High
    else:
        return 3  # Very high


df["PR_Val"] = df["PR_Val"].apply(assign_mews_score_to_pulse_rate)
df["Resp_Val"] = df["Resp_Val"].apply(assign_mews_score_to_respiratory_rate)
df["SPO2"] = df["SPO2"].apply(assign_mews_score_to_spo2)

for i in df.columns:
    df[i] = df[i].replace('-', 'N')

df['pattern'] = df.apply(lambda row: ''.join(map(str, row[1:])), axis=1)

pmews = df[['Time', 'pattern']]

pmews.columns = ['Time', 'pattern']

# pmews

print(pmews)

pattern_length = 5
pmews['trust'] = pmews['pattern'].apply(lambda x:((pattern_length-x.count('N'))/pattern_length)*100)

print(pmews.head())

pmews["pattern"].value_counts()


