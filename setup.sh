#!/bin/bash

# needed for parent project scripts to run (`mpe_hierarchy` as submodule)
# all import from `mpe_hierarchy` start imports with `multiagent`
export PYTHONPATH=$(pwd):$PYTHONPATH
