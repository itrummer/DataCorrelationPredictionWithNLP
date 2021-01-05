for coef in pearson spearman theilsu; do
	for mod in "roberta roberta-base" "bert bert-base-cased" "distilbert distilbert-base-cased" "albert albert-base-v2" "xlm xml-mlm-en-2048"; do
		for scenario in defsep; do
			echo $coef $mod $scenario;
			python3 coefficients.py $coef 0.5 1 $mod $scenario &> $coef"$mod"$scenario.log
		done;
	done;
done
#python3 coefficients.py pearson 0.5 1 &> ac1.log
#python3 coefficients.py spearman 0.5 1 &> ac2.log
#python3 coefficients.py theilsu 0.5 1 &> ac3.log
#python3 coefficients.py pearson 0.5 0.05 &> ac4.log
#python3 coefficients.py spearman 0.5 0.05 &> ac5.log
#python3 coefficients.py theilsu 0.5 0.5 &> ac6.log

