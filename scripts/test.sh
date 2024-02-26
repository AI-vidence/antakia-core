#!/usr/bin/env sh

poetry run python -m pytest --cov=antakia_core --cov-report term-missing tests
