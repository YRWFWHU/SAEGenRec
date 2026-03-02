#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = saegenrec
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	pip install -e .
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) saegenrec/dataset.py

CONFIG ?= configs/default.yaml

## Run data processing pipeline (legacy, all steps)
.PHONY: data-process
data-process:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG)

## Stage 1: load → filter → sequence
.PHONY: data-filter
data-filter:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) --step load --step filter --step sequence

## Stage 2: split → augment → negative_sampling
.PHONY: data-split
data-split:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) --step split --step augment --step negative_sampling

## Generate all embeddings (semantic + collaborative via pipeline)
.PHONY: data-embed
data-embed:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset process $(CONFIG) --step embed

## Generate semantic embeddings only
.PHONY: data-embed-semantic
data-embed-semantic:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset embed-semantic $(CONFIG)

## Generate collaborative embeddings only
.PHONY: data-embed-collaborative
data-embed-collaborative:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset embed-collaborative $(CONFIG)

## Tokenize items (embedding → SID map)
.PHONY: data-tokenize
data-tokenize:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset tokenize $(CONFIG)

## Build SFT instruction-tuning dataset
.PHONY: data-build-sft
data-build-sft:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset build-sft $(CONFIG)

## Download item images
.PHONY: data-download-images
data-download-images:
	$(PYTHON_INTERPRETER) -m saegenrec.dataset download-images configs/default.yaml


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
