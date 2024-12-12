#!/usr/bin/bash
 
echo hello-world

python -u -m run.INCL_RA
python -u -m run.INCL_CA_Max
python -u -m run.INCL_CA_Min
python -u -m run.INCL_RD
python -u -m run.INCL_CD_Max
python -u -m run.INCL_CD_Min
echo finished