#!/bin/bash
#
#SBATCH --job-name=one_year_lisa
#SBATCH --output=/fred/oz980/avajpeyi/projects/pywavelet/gaps_study/docs/one_year/logs/job_%A_%a.out
#SBATCH --error=/fred/oz980/avajpeyi/projects/pywavelet/gaps_study/docs/one_year/logs/job_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

ml python-scientific/3.11.3-foss-2023a
source /fred/oz980/avajpeyi/projects/pywavelet/venv/bin/activate

readarray -t jobs < '/fred/oz980/avajpeyi/projects/pywavelet/gaps_study/docs/one_year/slurm_jobs.txt'

echo "job cmd: ${jobs[$SLURM_ARRAY_TASK_ID]}"
eval "${jobs[$SLURM_ARRAY_TASK_ID]}"