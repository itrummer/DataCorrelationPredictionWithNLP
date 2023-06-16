create table predictions(rowid int, rowid2 int, dataid int, datapath text, nrrows numeric, nrvals1 numeric, nrvals2 numeric, type1 text, type2 text, column1 text, column2 text, method text, coefficient numeric, pvalue numeric, time numeric, labels numeric, ptime numeric, predictions numeric, jaccard numeric);

\copy predictions from 'theilsu.csv' delimiter ',' CSV header;

with scope as (select * from predictions where method = 'pearson'), percentiles as (select k, percentile_disc(k) within group (order by P.predictions) from scope P, generate_series(0.01, 1, 0.01) as k group by k) select round(100*k), avg(coefficient*coefficient) from scope S, percentiles P where predictions > P.percentile_disc group by k;