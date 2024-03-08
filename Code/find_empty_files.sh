#!/bin/bash

for file in RFLR*.err; do
    if [ ! -s $file ]
    then
        NAME=${file%.*}
        rm $file  #echo "$file is empty"
        rm "$NAME.out"
    else
        echo "$file is not empty"
    fi
done
