PAPER := IEEE-conference-template-062824
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all pdf rebuild clean

all: pdf

pdf:
	$(LATEXMK) $(LATEXMK_FLAGS) $(PAPER).tex

rebuild:
	$(LATEXMK) $(LATEXMK_FLAGS) -g $(PAPER).tex

clean:
	$(LATEXMK) -C $(PAPER).tex
