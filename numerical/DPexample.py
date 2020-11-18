# %%
# A example code for Dynamic Programming
# use VFI to solve the Bellman equation

import numpy
import pandas

# parameters in Bellman equation
A = 1/3
B = 0.96
# nodes
M = 50
# stop rule setting (difference of v between one loop)
D = 0.1

# function v()
v = lambda s, s1, v1: (-(s**A-s1)**(-1) + B*v1)
# initial value
v_ini = -100

# the dataframe for computation
# s_t
df = pandas.DataFrame({'s': numpy.linspace(0, 1, M+1)})
# v(s_t)
df['v'] = v_ini
# s_{t+1}
df['s1'] = numpy.NaN
# v(s_{t+1})
df['v1'] = numpy.NaN
# v(s_{t+1}) - v(s_t)
df['d'] = 999

# %%
# VFI
time = 1
while abs(df['d']).max() > D:
    df['v1'] = df['v']
    for j in range(0, M+1):
        s = df['s'][j]
        v_j = df[['s']].copy(deep=True)
        v_j['v'] = numpy.NaN
        for n in range(0, M+1):
            if (s**A - df['s'][n]) >= 0:
                v_j['v'][n] = v(s, df['s'][n], df['v1'][n])
        df['v'][j] = v_j[-numpy.isinf(v_j['v'])]['v'].max()
        df['s1'][j] = (v_j[v_j['v'] == df['v'][j]]['s'] if
                       not numpy.isnan(df['v'][j]) else numpy.NaN)
    df['d'] = abs(df['v'] - df['v1'])
    time = time + 1

# %%
df
