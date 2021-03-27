default:
	@echo "These are global shortcuts to build AID"
	@echo "Usage:"
	@echo "\tmake format\t Format all code"

format:
	autoflake --in-place --remove-unused-variables --recursive .
	isort .
	yapf -ir .

test:
	python3 -m unittest tests/*.py

package:
	python3 setup.py sdist bdist_wheel

upload:
	python3 -m twine upload --repository pypi dist/*