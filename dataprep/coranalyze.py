import pandas as pd
import sys

# get command line params
path = sys.argv[1]
dsid = sys.argv[2]

# read and factorize data
data = pd.read_csv(path)
fdata = data.apply(lambda x: x.factorize()[0])

# get basic info on columns
types = data.dtypes
nr_cols = len(data.columns)

# calculate column stats
nr_vals = []
for c in range(0, nr_cols):
    counts = data.iloc[:,c].value_counts()
    nr_vals.append(len(counts))
nr_rows = len(data.index)

# iterate over correlation metrics
for m in ['pearson', 'kendall', 'spearman']:
    cor = fdata.corr(method=m)
    cols = cor.columns
    for i in range(0, nr_cols):
        for j in range(0, i):
            col1 = cols[i]
            col2 = cols[j]
            type1 = types[i]
            type2 = types[j]
            nrv1 = nr_vals[i]
            nrv2 = nr_vals[j]
            cur_cor = cor.iloc[i,j]
            print(f'{dsid},{path},{m},{nr_rows},{nrv1},{nrv2},{type1},{type2},{col1},{col2},{cur_cor}')

