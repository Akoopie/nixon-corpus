# Build the paper.
#   make           latexmk + pdflatex + bibtex (TeX Live: apt install texlive-full latexmk)
#   make tectonic  single-binary build (https://tectonic-typesetting.github.io); fetches packages on first run
#   make arxiv     source tarball for arXiv upload (includes the .bbl that arXiv needs)
#   make clean
MAIN = main

all:
	latexmk -pdf -interaction=nonstopmode $(MAIN).tex

tectonic:
	tectonic --keep-intermediates $(MAIN).tex

arxiv: all
	tar czf nixon-corpus-arxiv.tar.gz $(MAIN).tex $(MAIN).bbl metadata.tex references.bib sections figures/*.pdf

clean:
	latexmk -C $(MAIN).tex
	rm -f $(MAIN).bbl $(MAIN).blg nixon-corpus-arxiv.tar.gz
