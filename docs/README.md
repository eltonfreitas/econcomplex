# Documentation

This folder stores the English and Portuguese technical documentation for `econcomplex`.

- English PDF: [econcomplex_documentation_en.pdf](econcomplex_documentation_en.pdf)
- Portuguese PDF: [econcomplex_documentation_pt.pdf](econcomplex_documentation_pt.pdf)
- English LaTeX source: [econcomplex_documentation_en.tex](econcomplex_documentation_en.tex)
- Portuguese LaTeX source: [econcomplex_documentation_pt.tex](econcomplex_documentation_pt.tex)

## Rebuilding

Two parts of the PDFs are generated from the package, so that they cannot
drift from the code. Run both from the repository root after changing the
public API, then recompile the two `.tex` files:

```bash
PYTHONPATH=. python docs/generate_api_reference.py   # api_reference_{en,pt}.tex
PYTHONPATH=. python docs/generate_tutorial.py        # tutorial_{en,pt}.tex and examples/tutorial.py
latexmk -pdf -outdir=build docs/econcomplex_documentation_en.tex
latexmk -pdf -outdir=build docs/econcomplex_documentation_pt.tex
```

The tutorial generator needs scikit-learn (`pip install "econcomplex[ml]"`).
It stops with an error if the output of a step no longer supports what the
text of that step says.
