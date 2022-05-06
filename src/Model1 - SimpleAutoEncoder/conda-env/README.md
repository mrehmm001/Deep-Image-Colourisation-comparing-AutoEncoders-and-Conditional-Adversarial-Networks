# Using `conda` in Slurm

If you need to install particular packages the easiest way is to
create a dedicated conda environmnt.

When you submit your job, it gets ran on an entirely different server
(`lara.doc.gold.ac.`). This server mounts your same home directory,
but has a completey different runtime environment. Therefore, your
conda environment needs to be set up from within `lara` – meaning that
you need to submit your environment setup code as a slurm job.

Run the follwing scripts to create, activate and remove enviroments.

Note, you must activate the conda environmnet in the slurm script for
every job you want to run in your conda environment.

First, make sure that `conda` is on your $PATH by adding the following
to your `~/.bashrc`.

``` conf
export PATH=$PATH:/usr/local/anaconda3/bin
```


``` sh
sbatch conda-env-create.sh
tail -f conda-env.log          # Watch the log.

sbatch test-job.sh
tail -f test.log               # Prints Python version in the conda env.

sbatch mnist-conda.job.sh      # Run mnist inside your conda env.
tail -f mnist.log

sbatch conda-env-remove.sh     # Remove your conda env.
tail -f conda-env.log
```
