dependencies:
	Rscript dependencies.R

data: dependencies
	Rscript movies.R

repro: data
	(R CMD BATCH --vanilla code.R &)

all: dependencies data repro

clean:
	rm -rf *.RDS *.pdf *.Rout
