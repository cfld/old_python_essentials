#!/bin/bash

# build.sh

# --
# Build

rm -rf build
mkdir build
cd build
cmake ..
make -j12
cd ..

# --
# Run

python test.py
