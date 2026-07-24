#! /bin/bash

for BAND in 4 5
do
    echo "Executando BAND $BAND  - bash"
    # python Espectral/01_Pipeline_Espectral.py --BAND $BAND
    python Espectral/01_Model_Espectral.py --BAND $BAND
done

echo "bash done"

