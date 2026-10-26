# The Nixon Corpus: paper source (arXiv)

## Editing

| To change | Edit |
|---|---|
| Title, author, ORCID, version line, repository/model/DOI links | `metadata.tex` (top of the file) |
| A section's text | `sections/NN_name.tex`, one file per section |
| References | `references.bib` |
| Figures | regenerate with `python paper/figure_data.py && python paper/speaker_time.py && python paper/make_figures.py && python paper/make_fig1.py`, then copy `paper/figures/fig*.pdf` here |

Helpers defined in `main.tex`: `\conv{871-004}` for conversation numbers and `\machine{...}` for quoted machine
transcript. Every `\machine{}` quotation is checked against the corpus by `python paper/check_quotes.py`.

## Building

On Linux with TeX Live (`sudo apt install texlive-latex-extra texlive-fonts-extra texlive-science latexmk`):

```bash
make            # latexmk + pdflatex + bibtex -> main.pdf
make arxiv      # nixon-corpus-arxiv.tar.gz for upload (includes main.bbl, which arXiv needs)
make clean
```

Or with the single-binary [Tectonic](https://tectonic-typesetting.github.io) engine, which downloads packages on first
use: `make tectonic`.
