PAPER := IEEE-conference-template-062824
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error
FIGURE_NAMES := reference_transfer_plot matched_localization_plot harm_specificity_plot overlap_profiles process_dose_plot measured_load_plot
FIGURE_FRAGMENTS := $(addprefix figures/,$(addsuffix .tex,$(FIGURE_NAMES)))
FIGURE_PDFS := $(addprefix figures/,$(addsuffix _standalone.pdf,$(FIGURE_NAMES)))

.PHONY: all pdf figures rebuild clean FORCE

all: pdf

pdf:
	$(LATEXMK) $(LATEXMK_FLAGS) $(PAPER).tex

figures: figures/publication_figures.pdf $(FIGURE_PDFS)

# Invoke latexmk even when an extracted archive supplies a newer prebuilt PDF.
# Its own incremental database avoids unnecessary TeX runs in a live checkout;
# a clean extraction rebuilds the missing logs required by PDF verification.
figures/publication_figures.pdf: figures/publication_figures.tex figures/publication_style.tex $(FIGURE_FRAGMENTS) FORCE
	cd figures && $(LATEXMK) $(LATEXMK_FLAGS) publication_figures.tex

figures/%_standalone.pdf: figures/%_standalone.tex figures/%.tex figures/publication_style.tex FORCE
	cd figures && $(LATEXMK) $(LATEXMK_FLAGS) $(<F)

rebuild:
	$(LATEXMK) $(LATEXMK_FLAGS) -g $(PAPER).tex

clean:
	$(LATEXMK) -C $(PAPER).tex

FORCE:
