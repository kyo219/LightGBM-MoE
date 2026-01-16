#!/bin/bash

set -e -E -u -o pipefail

conda create -q -y -n test-env \
    "python=3.13[build=*_cp*]" \
    'pre-commit>=3.8.0'

# shellcheck disable=SC1091
source activate test-env

echo "Running pre-commit checks"
pre-commit run --all-files || exit 1
