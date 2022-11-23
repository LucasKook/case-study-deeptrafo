dependencies:
	Rscript dependencies.R

data: dependencies
	Rscript movies.R

repro: data
	Rscript code.R

all: dependencies data repro

clean:
	rm -rf *.RDS *.pdf
