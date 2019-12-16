#!/usr/bin/env bash

# Check if user passed in a list of species, else use the default.
if [ $# -eq 0 ]; then
    declare -a SPECIES_ARR=('caenorhabditis_elegans' 'drosophila_melanogaster' 'danio_rerio' 'mus_musculus')
else
    declare -a SPECIES_ARR=( "$@" )
fi

# Download weights for each species.
for SPECIES in "${SPECIES_ARR[@]}"; do
    wget 'http://deepark.princeton.edu/media/code/'"${SPECIES}"'.pth.tar' --directory-prefix=data
    if [ $? != 0 ]; then
        echo 'Failed to download weights for '"${SPECIES}"
        exit 1
    fi
done

