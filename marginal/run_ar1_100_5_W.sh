#!/bin/bash
#SBATCH --job-name=run_ar1_100_5_W
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=128gb
#SBATCH --time=24:00:00
#SBATCH --output=run_ar1_100_5_W.out


module load R/4.0.5

# run R

#Rscript /flush2/li042/BB_Genotype_file/RF_Ruby.R
Rscript run_ar1_100_5_W.R
