#!/bin/sh

set -Ceux

: ${PYTHON:=python}

root=$(cd -- "$(dirname -- "$0")" && pwd)
cd -- "${root}"

rm -rf htmlcov

python -m coverage run --source=src/ -m pytest
python -m coverage html
python -m coverage report
