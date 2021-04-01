#!/bin/bash
#SBATCH --partition=main                      # Ask for unkillable job
#SBATCH --cpus-per-task=4                     # Ask for 2 CPUs
#SBATCH --gres=gpu:2                          # Ask for 1 GPU
#SBATCH --mem=16G                             # Ask for 10 GB of RAM
#SBATCH --time=23:59:00                        # The job will run for 3 hours
#SBATCH -o /network/projects/s/soroosh.shahtalebi/slurm-%j.out  # Write the log on tmp1

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
# conda activate <env_name>
module load pytorch
module load tensorflow
source $HOME/env-ILC/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/cifar10 $SLURM_TMPDIR
#cp /network/datasets/cifar10 /tmp/cifar_dataset
# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
# python main.py --path $SLURM_TMPDIR --data_path $SLURM_TMPDIR

#python -m and_mask.run_synthetic --method=and_mask --agreement_threshold=1. --n_train_envs=16 --n_agreement_envs=16 --batch_size=256 --n_dims=16 --scale_grad_inverse_sparsity=1 --use_cuda=1 --n_hidden_units=256
cd ./ILC/code

python -m and_mask.run_cifar --random_labels_fraction 1.0 --agreement_threshold 0.2 --method and_mask --epochs 80 --weight_decay 1e-06 --scale_grad_inverse_sparsity 1 --init_lr 0.0005 --weight_decay_order before --output_dir ./tmp/


#for i in {0..50}
#do
#    python -m and_mask.run_synthetic --method=geom_mean --agreement_threshold=.8 --n_train_envs=4 --n_agreement_envs=4 --batch_size=256 --#n_dims=16 --scale_grad_inverse_sparsity=1 --use_cuda=1 --n_hidden_units=256 --seed=$i
#done




# 5. Copy whatever you want to save on $SCRATCH
#cp $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
