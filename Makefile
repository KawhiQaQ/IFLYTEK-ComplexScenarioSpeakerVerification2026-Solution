PYTHON ?= python
INPUT ?=
OUTPUT ?= outputs/inference
WEIGHTS ?= weights

.PHONY: bootstrap install-weights verify plan smoke train infer

bootstrap:
	bash scripts/bootstrap.sh

install-weights:
	$(PYTHON) scripts/install_weights.py --weights "$(WEIGHTS)"

verify:
	$(PYTHON) scripts/verify_assets.py
	$(PYTHON) -m compileall -q src scripts

plan:
	$(PYTHON) scripts/train.py --dry-run

smoke:
	$(PYTHON) scripts/train.py --stage prototypes
	$(PYTHON) scripts/train.py --stage wide-residual --smoke
	$(PYTHON) scripts/train.py --stage cross-encoder --smoke

train:
	$(PYTHON) scripts/train.py

infer:
	@test -n "$(INPUT)" || (echo 'Set INPUT=/path/to/official/input' >&2; exit 2)
	$(PYTHON) scripts/infer.py --input "$(INPUT)" --output "$(OUTPUT)"
