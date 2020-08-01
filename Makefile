default:
	@echo "These are global shortcuts to build AID"
	@echo "Usage:"
	@echo "\tmake format\t Format all code"

format:
	autoflake -i **/*.py
	isort -i **/*.py
	yapf -i **/*.py

test:
	python3 -m unittest tests/*.py
