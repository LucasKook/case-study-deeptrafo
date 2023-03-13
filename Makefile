dependencies:
	cd Code && Rscript dependencies.R

data: dependencies
	cd Code && Rscript movies.R

repro: data
	(cd Code && R CMD BATCH --vanilla code.R &)

all: dependencies data repro

clean:
	rm -rf **/*.RDS **/*.pdf **/*.Rout Figures
