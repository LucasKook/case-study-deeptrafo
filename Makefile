dependencies:
	cd Code && Rscript dependencies.R

data: 
	cd Code && Rscript movies.R

repro: 
	(cd Code && R CMD BATCH --vanilla code.R &)

all: dependencies data repro

clean:
	rm -rf **/*.RDS **/*.pdf **/*.Rout Figures
