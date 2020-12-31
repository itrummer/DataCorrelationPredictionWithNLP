find ../data -name '*.csv' > ../data/allcsvs
rm ../data/corresult2.csv
let 'x=0'; while read f; do let 'x=x+1'; python3 coranalyze.py "$f" $x >> ../data/corresult4.csv; done < ../data/allcsvs
