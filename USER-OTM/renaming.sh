#!/bin/bash

# How to rename multiple files from smd to otm
for file in *smd*; do
mv "$file" "${file//smd/otm}"
done

