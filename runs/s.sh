#!/bin/bash -l

# `-l`: make bash act as if it had been invoked as a login shell:
# The argument is indispensable for activating commands `.bashrc`
# L'argument '-l' est indispensable pour bénéficier des directives de votre .bashrc

# Here placed the commands SBATCH defining the parameters for node reservation
# On peut éventuellement placer ici les commentaires SBATCH permettant de
# définir les paramètres par défaut de lancemen:

#SBATCH --time 2-00:00:00
#SBATCH --exclude=sn[2-7]#,sw1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mail-type=FAIL,END
#SBATCH --mem-per-cpu=3G

# Spécification de la version CUDA
# setcuda 11.6

# Activation de l'environnement anaconda python37

conda activate bmvc22
# Exécution du script habituellement utilisé, on utilise la variable
# CUDA_VISIBLE_DEVICES qui contient la liste des GPU logiques actuellement réservé
export PYTHONPATH=/share/home/leh/anaconda3/envs/bmvc22/lib/python3.9/site-packages/pycocotools/:$PYTHONPATH

params="${@:1}"

# with mutual + gflocal loss
export WANDB_MODE=online
export WANDB_PROJECT=tw_eccv24

for kdc in "soft"; do # "hard"; do
    for kdf in "" "+mse" ; do #"+pdf" "+defeat"; do

        kd=${kdc}${kdf}
        # ------------------------
        # TODO: to be removed
        export WANDB_MODE=offline
        # TODO: to be changed to sbatch runs/distil.sh
        # ------------------------
        python distil.py    --seed 0 --eval_epoch 1 --batch_size 10 \
                            --kd ${kd} ${params}
    done
done
