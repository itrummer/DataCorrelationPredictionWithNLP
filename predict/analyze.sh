for coef in pearson spearman theilsu; do
	for mod in "roberta roberta-base" "distilbert distilbert-base-cased" "albert albert-base-v2"; do
		for scenario in defsep datasep; do
			for limits in "0.25 1" "0.5 1" "0.75 1" "0.5 0.05"; do 
				for testratio in "0.2" "0.8"; do
					echo $coef $mod $scenario $limits $testratio
					python3 coefficients.py $coef $limits $mod $scenario $testratio
				done;
			done;
		done;
	done;
done

