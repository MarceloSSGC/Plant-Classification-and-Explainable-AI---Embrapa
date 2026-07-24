#!/bin/bash

for SEED in 70
do
    for ALIGN in 0 1
    do 
        printf '\n'
        printf '=%.0s' {1..80}
        echo -e "\n BASH::  SEED: $SEED - ALIGN: $ALIGN"
       
        python Model_02/02_Pipeline__No_Repetition.py --SEED $SEED --ALIGN $ALIGN 
        python Model_02/02_Model__No_Repetition.py --SEED $SEED --ALIGN $ALIGN 
    done
done

printf '\n\n\n'
printf '=%.0s' {1..22}
echo -e "\n --- Finazilado ---"
printf '=%.0s' {1..22}
printf '\n'
