# pyArgos Makefile
# Usage: make <target>

PYARGOS_ROOT := $(shell pwd)
PYTHON       ?= python3
VENV_NAME    ?= Argos

.PHONY: help install install-deps install-dev env env-persist \
        test test-verbose \
        mermaid-pull render-diagrams render-diagrams-force \
        docs-deps docs-build docs-build-strict docs-serve docs-deploy docs-clean

help:
	@echo "pyArgos targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install             Full install (deps + env vars + prompt for .bashrc)"
	@echo "    make install-deps        Install Python dependencies from requirements.txt"
	@echo "    make install-dev         Install dev + optional dependencies (Kafka, TB, docs)"
	@echo "    make env                 Print the environment variables needed for pyArgos"
	@echo "    make env-persist         Add PYTHONPATH to ~/.bashrc (prompts for confirmation)"
	@echo ""
	@echo "  Testing:"
	@echo "    make test                Run all tests"
	@echo "    make test-verbose        Run all tests with full output"
	@echo ""
	@echo "  Documentation:"
	@echo "    make docs-build          Build full site (render diagrams + mkdocs build)"
	@echo "    make docs-serve          Start local MkDocs preview server"
	@echo "    make docs-deploy         Deploy to GitHub Pages"
	@echo "    make docs-clean          Remove built site/"
	@echo "    make render-diagrams     Render new/changed Mermaid diagrams to SVG"
	@echo "    make render-diagrams-force  Re-render ALL diagrams"
	@echo "    make mermaid-pull        Pull the mermaid-cli Docker image"
	@echo "    make docs-deps           Install documentation dependencies"

# --- Setup & Installation ---

install: install-deps env-persist
	@echo ""
	@echo "=== pyArgos installed ==="
	@echo "  Activate with: source ~/.bashrc  (or open a new terminal)"
	@echo "  Verify with:   $(PYTHON) -c 'import argos; print(argos.__version__)'"
	@echo "  Docs:          https://argosp.github.io/pyargos/"

install-deps:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Done."

install-dev: install-deps
	@echo "Installing optional/dev dependencies..."
	pip install kafka-python tb_rest_client mkdocs-material mkdocstrings mkdocstrings-python
	@echo "Done."

env:
	@echo ""
	@echo "=== pyArgos environment variables ==="
	@echo ""
	@echo "Add these to your shell profile (~/.bashrc or ~/.zshrc):"
	@echo ""
	@echo "  export PYTHONPATH=\"$(PYARGOS_ROOT):\$$PYTHONPATH\""
	@echo ""
	@echo "Current PYTHONPATH: $${PYTHONPATH:-<not set>}"
	@echo ""
	@if echo "$$PYTHONPATH" | grep -q "$(PYARGOS_ROOT)"; then \
		echo "  ✓ pyArgos is already in PYTHONPATH"; \
	else \
		echo "  ✗ pyArgos is NOT in PYTHONPATH"; \
		echo "    Run 'make env-persist' to add it to ~/.bashrc"; \
	fi

env-persist:
	@SHELL_RC="$$HOME/.bashrc"; \
	if [ -f "$$HOME/.zshrc" ] && [ "$$SHELL" = "/bin/zsh" -o "$$SHELL" = "/usr/bin/zsh" ]; then \
		SHELL_RC="$$HOME/.zshrc"; \
	fi; \
	EXPORT_LINE="export PYTHONPATH=\"$(PYARGOS_ROOT):\$$PYTHONPATH\"  # pyArgos"; \
	echo ""; \
	if grep -q "# pyArgos" "$$SHELL_RC" 2>/dev/null; then \
		echo "pyArgos is already configured in $$SHELL_RC:"; \
		grep "# pyArgos" "$$SHELL_RC"; \
		echo ""; \
		echo "No changes made."; \
	else \
		echo "This will add the following line to $$SHELL_RC:"; \
		echo ""; \
		echo "  $$EXPORT_LINE"; \
		echo ""; \
		read -p "Add to $$SHELL_RC? [y/N] " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			echo "" >> "$$SHELL_RC"; \
			echo "$$EXPORT_LINE" >> "$$SHELL_RC"; \
			echo ""; \
			echo "Added to $$SHELL_RC"; \
			echo "Run 'source $$SHELL_RC' or open a new terminal to activate."; \
		else \
			echo ""; \
			echo "Skipped. To set manually, run:"; \
			echo "  echo '$$EXPORT_LINE' >> $$SHELL_RC"; \
		fi; \
	fi

# --- Testing ---

test:
	pytest

test-verbose:
	pytest -v --tb=long

# --- Documentation ---

mermaid-pull:
	@echo "Pulling mermaid-cli Docker image..."
	docker pull minlag/mermaid-cli
	@echo "Done. Run 'make render-diagrams' to render all Mermaid diagrams."

render-diagrams: mermaid-pull
	@echo "Rendering Mermaid diagrams to SVG (skips unchanged)..."
	python render_diagrams.py
	@echo "Done. SVGs saved to docs/images/diagrams/"

render-diagrams-force: mermaid-pull
	@echo "Re-rendering ALL Mermaid diagrams to SVG..."
	python render_diagrams.py --force
	@echo "Done. All SVGs regenerated."

docs-deps:
	@echo "Installing documentation dependencies..."
	pip install mkdocs-material mkdocstrings mkdocstrings-python
	@echo "Done."

docs-build: render-diagrams
	@echo "Building MkDocs site..."
	mkdocs build
	@echo "Done. Static site built in site/"

docs-build-strict: render-diagrams
	@echo "Building MkDocs site (strict mode)..."
	mkdocs build --strict
	@echo "Done. Static site built in site/"

docs-serve: render-diagrams
	@echo "Starting MkDocs development server at http://127.0.0.1:8000"
	./serve_docs.sh

docs-deploy: render-diagrams
	@echo "Deploying documentation to GitHub Pages..."
	mkdocs gh-deploy --force
	@echo "Done. Site deployed."

docs-clean:
	@echo "Removing built site..."
	rm -rf site/
	@echo "Done."
