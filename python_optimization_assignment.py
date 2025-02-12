# -*- coding: utf-8 -*-
"""Python Optimization Assignment

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16W_RzhFMbR9fVFIsUADb10eMdpslQZdH
"""

import pandas as pd
import numpy as np
from math import *

df = pd.read_csv('clinics.csv', delimiter="|", encoding="utf-8")

df.head()

print(df.columns)

"""**Define the normalization function**"""

def normalize(df, column_name):
    pd_series = df[column_name].astype(float)

# Find upper and lower bound for outliers
    avg = np.mean(pd_series)
    sd = np.std(pd_series)
    lower_bound = avg - 2 * sd
    upper_bound = avg + 2 * sd

 # Collapse in the outliers
    df.loc[df[column_name] < lower_bound, "cutoff_rate"] = lower_bound
    df.loc[df[column_name] > upper_bound, "cutoff_rate"] = upper_bound

# Finally take the log
    df["normalized_value"] = np.log(df["cutoff_rate"].replace(0, np.nan))  # Avoid log(0) issues

    return df["normalized_value"]

"""Timing the normalization function"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit df['locLat_normalized'] = normalize(df, 'locLat')

"""Profiling the normalization function"""

!pip install line-profiler

# Commented out IPython magic to ensure Python compatibility.
# %load_ext line_profiler

# Commented out IPython magic to ensure Python compatibility.
# %lprun -f normalize normalize(df, "locLat")

def haversine(lat1, lon1, lat2, lon2):
    miles_constant = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    mi = miles_constant * c
    return mi

import time

# Start timer
start_time = time.time()

# Slowest approach: Using iterrows()
haversine_series = []
for index, row in df.iterrows():
    haversine_series.append(haversine(40.671, -73.985, row['locLat'], row['locLong']))

df['distance'] = haversine_series

"""**Apply Haversine on rows**

Timing "apply"
"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit df['distance'] = df.apply(lambda row: haversine(40.671, -73.985, row['locLat'], row['locLong']), axis=1)

"""Profiling "apply"
"""

# Commented out IPython magic to ensure Python compatibility.
# Haversine applied on rows
# %lprun -f haversine \
df.apply(lambda row: haversine(40.671, -73.985,\
                               row['locLat'], row['locLong']), axis=1)

"""**Vectorized implementation of Haversine applied on Pandas series**

Timing vectorized implementation
"""

# Commented out IPython magic to ensure Python compatibility.
# Vectorized implementation of Haversine applied on Pandas series
# %timeit df['distance'] = haversine(40.671, -73.985,\
                                   df['locLat'], df['locLong'])

"""Profiling vectorized implementation"""

# Commented out IPython magic to ensure Python compatibility.
# Vectorized implementation profile
# %lprun -f haversine haversine(40.671, -73.985,\
                              df['locLat'], df['locLong'])

"""**Vectorized implementation of Haversine applied on NumPy arrays**

Timing vectorized implementation
"""

# Commented out IPython magic to ensure Python compatibility.
# Vectorized implementation of Haversine applied on NumPy arrays
# %timeit df['distance'] = haversine(40.671, -73.985,\
                         df['locLat'].values, df['locLong'].values)

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# # Convert pandas arrays to NumPy ndarrays
# np_lat = df['locLat'].values
# np_lon = df['locLong'].values

"""Profiling vectorized implementation"""

# Commented out IPython magic to ensure Python compatibility.
# %lprun -f haversine df['distance'] = haversine(40.671, -73.985,\
                        df['locLat'].values, df['locLong'].values)

"""**Cythonize that loop**

Load the cython extension
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext cython

"""Run unaltered Haversine through Cython

"""

# Commented out IPython magic to ensure Python compatibility.
# %%cython -a
# 
# # Haversine cythonized (no other edits)
# import numpy as np
# cpdef haversine_cy(lat1, lon1, lat2, lon2):
#     miles_constant = 3959
#     lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     mi = miles_constant * c
#     return mi

"""Time it"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit df['distance'] =\
       df.apply(lambda row: haversine_cy(40.671, -73.985,\
                row['locLat'], row['locLong']), axis=1)

"""Redefine Haversine with data types and C libraries"""

# Commented out IPython magic to ensure Python compatibility.
# %%cython -a
# # Haversine cythonized
# from libc.math cimport sin, cos, acos, asin, sqrt
# 
# cdef deg2rad_cy(float deg):
#     cdef float rad
#     rad = 0.01745329252*deg
#     return rad
# 
# cpdef haversine_cy_dtyped(float lat1, float lon1, float lat2, float lon2):
#     cdef:
#         float dlon
#         float dlat
#         float a
#         float c
#         float mi
# 
#     lat1, lon1, lat2, lon2 = map(deg2rad_cy, [lat1, lon1, lat2, lon2])
#     dlat = lat2 - lat1
#     dlon = lon2 - lon1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     mi = 3959 * c
#     return mi

"""Time it

"""

# Commented out IPython magic to ensure Python compatibility.
# %timeit df['distance'] =\
df.apply(lambda row: haversine_cy_dtyped(40.671, -73.985,\
                              row['locLat'], row['locLong']), axis=1)