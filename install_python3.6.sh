#!/bin/bash

PY_VERSION="3.6"
PY_DIR="${HOME}/python${PY_VERSION}"
PY_EXE="${PY_DIR}/bin/python${PY_VERSION}"
VIRTUALENV="${HOME}/virtualenv"

rm -rf ${PY_DIR}
mkdir -p ${PY_DIR}
cd ${PY_DIR}

git clone https://github.com/python/cpython.git git
cd git
git checkout ${PY_VERSION}

./configure --enable-optimizations --prefix=${PY_DIR} --exec-prefix=${PY_DIR}
make altinstall prefix=${PY_DIR} exec-prefix=${PY_DIR}

${PY_EXE} -m venv "${VIRTUALENV}" --clear

echo "Activate virtual environment with:"
echo source "${VIRTUALENV}/bin/activate"
echo "source \"${VIRTUALENV}/bin/activate\"" >> ~/.bashrc



