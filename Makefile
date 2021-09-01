# Makefile for Sphinx documentation
#

SPHINXBUILD ?= sphinx-build
SOURCEDIR    = docs
BUILDDIR     = docs/build

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

.PHONY: help Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)"
