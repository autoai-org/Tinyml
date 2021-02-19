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

package:
	python3 setup.py sdist bdist_wheel

upload:
	python3 -m twine upload --repository pypi dist/*