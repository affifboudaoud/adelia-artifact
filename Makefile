PAPER=artifact_description

$(PAPER).pdf: $(PAPER).tex sc26repro.sty acmart.cls
	pdflatex $(PAPER)
	pdflatex $(PAPER)
	pdflatex $(PAPER)

# `tectonic $(PAPER).tex` is a single-command drop-in replacement if
# pdflatex is unavailable.

clean:
	rm -f *.aux *.fls *.bbl *.blg *.log *.out *.fdb* *.synctex* *converted*.pdf

.PHONY: clean
