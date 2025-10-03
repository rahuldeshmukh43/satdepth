#!/bin/bash

# clone loftr
mkdir -p ../../external/LoFTR
git clone https://github.com/zju3dv/LoFTR.git ../../external/LoFTR

cd ../../external/LoFTR
# checkout a specific commit
git checkout 94e98b695be18acb43d5d3250f52226a8e36f839
cd -