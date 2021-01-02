rm datasets.txt
for page in {1..100}; do kaggle datasets list --file-type csv --max-size 1000000 -p $page --csv | tail -n 20 >> datasets.txt; echo $page; done

