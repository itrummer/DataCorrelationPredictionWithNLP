import pandas as pd
import scipy.stats as stats
import sys
import time
import dython.nominal as nom

# get command line params
path = sys.argv[1]
dsid = sys.argv[2]

# read, cut, and factorize data
#print(f'Reading data from path {path}', flush=True)
data = pd.read_csv(path)
nr_cols = len(data.columns)
if nr_cols > 10:
    data = data.iloc[:,range(0,10)]
#fdata = data.apply(lambda x: x.factorize()[0])

# get basic info on columns
types = data.dtypes
#print(f'Types: {types}', flush=True)
nr_cols = len(data.columns)
#print(f'Nr. columns: {nr_cols}', flush=True)

# calculate column stats
nr_vals = []
for c in range(0, nr_cols):
    counts = data.iloc[:,c].value_counts()
    nr_vals.append(len(counts))
nr_rows = len(data.index)
#print(f'Nr. rows: {nr_rows}')

# use all correlation metrics
def all_corr(row_pre, s1, s2):
    t1 = s1.dtype
    t2 = s2.dtype
    num_types = ['int64', 'float64']
    if t1 in num_types and t2 in num_types:
        s_time = time.time()
        r, p = stats.pearsonr(s1, s2)
        t_time = time.time() - s_time
        print(f'{row_pre}pearson,{r},{p},{t_time}')
        s_time = time.time()
        r, p = stats.spearmanr(s1, s2)
        t_time = time.time() - s_time
        print(f'{row_pre}spearman,{r},{p},{t_time}')
    s_time = time.time()
    u1 = nom.theils_u(s1, s2)
    u2 = nom.theils_u(s2, s1)
    t_time = time.time() - s_time
    print(f'{row_pre}theilsu,{u1},{u2},{t_time}')

# iterate over correlation metrics
cols = data.columns
for i in range(0, nr_cols):
    for j in range(0, i):
        col1 = cols[i]
        col2 = cols[j]
        type1 = types[i]
        type2 = types[j]
        nrv1 = nr_vals[i]
        nrv2 = nr_vals[j]
        row_pre = f'{dsid},"{path}",{nr_rows},{nrv1},{nrv2},' \
                f'{type1},{type2},"{col1}","{col2}",'
        c1_data = data[col1]
        c2_data = data[col2]
        # measure correlation and time
        all_corr(row_pre, c1_data, c2_data)

