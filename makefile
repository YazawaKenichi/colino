PROJECT = Project

MAIN = ./scripts/main.py
ALL = ./scripts/main.py ./scripts/*.py

RESULT = ./result.png
OUTPUT = ./*.png

PYTHON = python3
PYTOPT = 
EDITOR = vim
EDTOPT = -p

EDIT = edit
EDITING = editing

$(MAIN):
	$@

$(EDIT): $(EDITING) $(MAIN)

$(EDITING): $(ALL)
	$(EDITOR) $^ $(EDTOPT)

all: clean $(PROJECT)

clean:
	rm -rf $(RESULT)

clean_all:
	rm -rf $(OUTPUT)

