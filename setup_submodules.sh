#!/bin/bash
git submodule update --init --recursive
cd gitmodules/gpytorch && git checkout fc2053b && cd ../..
cd gitmodules/nsf && git checkout 8e3fe75 && cd ../..
cd gitmodules/BNAF && git checkout da43f56 && cd ../..

