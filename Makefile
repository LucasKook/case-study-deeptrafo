dependencies:
	Rscript dependencies.R

data:
	Rscript movies.R

repro: data
	Rscript code.R

