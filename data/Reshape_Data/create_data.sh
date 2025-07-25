#!/bin/bash
#
### Job Parameters:
# basic info
#SBATCH --job-name "process_data"               # name
#SBATCH --output "data-out.log"      # output file
#SBATCH --error "data-err.log"       # error message file

# resource request info 
#SBATCH --mem=32G

# Opt-into email alerts
#SBATCH --mail-type ALL
#SBATCH --mail-user hayavuzkara@davidson.edu

# This is environment created specifically for data, with things used in the three python files installed.
source /opt/conda/bin/activate data

# python process_mg_o_combo.py
python exp_process_var_in_len.py