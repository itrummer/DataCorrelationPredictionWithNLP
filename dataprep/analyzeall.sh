find . -name '*.csv' > allcsvs
let 'x=0'; while read f; do let 'x=x+1'; python3 coranalyze.py $f $x >> corresult2.csv; done < allcsvs
