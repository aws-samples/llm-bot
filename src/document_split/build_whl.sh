#!/bin/bash

cp ../lambda/embedding/utils/opensearch_vector_search.py dep/llm_bot_dep/
cp ../lambda/embedding/utils/sm_utils.py dep/llm_bot_dep/
cp ../lambda/embedding/build_index.py dep/llm_bot_dep/
cd ./dep
pip install setuptools wheel

python setup.py bdist_wheel