LATEX       = pdflatex
BIBTEX      = bibtex
BASH        = bash -c
ECHO        = echo
RM          = rm -rf
RM_TMP      = ${RM} $(foreach suff, ${TMP_SUFFS}, ${NAME}.${suff})
RMTEX       = rm -f *.dvi *.aux *.log *.blg *.out

TMP_SUFFS   = pdf aux blg log dvi ps eps out
SUFF        = pdf

CHECK_RERUN = grep Rerun $*.log

NAME    = paper
DOC_OUT = ${NAME}.${SUFF}

DEPENDENCIES = paper.tex intro.tex analysis.tex pipeline.tex discussion.tex  

default: ${DOC_OUT}

%.pdf: %.tex %.bib $(DEPENDENCIES)
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	${BIBTEX} $(<:.tex=.aux)
	${BIBTEX} $(<:.tex=.aux)
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done." 
clean:
	${RM_TMP}

