python3 coefficients.py pearson 0.5 1 &> ac1.log
python3 coefficients.py spearman 0.5 1 &> ac2.log
python3 coefficients.py theilsu 0.5 1 &> ac3.log
python3 coefficients.py pearson 0.5 0.05 &> ac4.log
python3 coefficients.py spearman 0.5 0.05 &> ac5.log
python3 coefficients.py theilsu 0.5 0.5 &> ac6.log

