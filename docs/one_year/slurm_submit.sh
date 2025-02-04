#!/bin/bash
#
#SBATCH --job-name=one_year_lisa
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=00:100:00
#SBATCH --array=0-3
#SBATCH --mem=6G
#SBATCH --cpus-per-task=4

ml gcc/12.3.0 numpy/1.25.1-scipy-bundle-2023.07 && source /fred/oz101/avajpeyi/compas_env/bin/activate

readarray -t gen_cmd < '/fred/oz101/avajpeyi/code/pipeline/studies/one_param/pp_test/out_aSF/gen_cmd.txt'
readarray -t analy_cmd < '/fred/oz101/avajpeyi/code/pipeline/studies/one_param/pp_test/out_aSF/analy_cmd.txt'

echo "Mock cmd: ${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
echo "Surrogate cmd: ${analy_cmd[$SLURM_ARRAY_TASK_ID]}"
eval "${gen_cmd[$SLURM_ARRAY_TASK_ID]}"
eval "${analy_cmd[$SLURM_ARRAY_TASK_ID]}"