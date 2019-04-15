# AttentionExplanation

This is code for the project : https://arxiv.org/abs/1902.10186 . 
We will be updating it in coming weeks to include instructions on how to download and process the data and run the experiments.

This project requires compiling `pytorch` from source master branch or use `pytorch-nightly`. We use features that are not in stable release.

Update
------

We are providing code to run experiments on all datasets except MIMIC for now (the latter requires access to MIMIC datasets)

1. Clone the repository as `git clone https://github.com/successar/AttentionExplanation.git Transparency` (Note this is important.)
2. Set your PYTHONPATH to include the directory name which contains this repository (All imports in the code are of form Transparency.*)
3. Go to the `Transparency/preprocess` folder and follow the instructions to process datasets.
4. From the main folder, run `python train_and_run_experiments_bc.py --dataset {dataset_name} --data_dir . --output_dir outputs/ --attention {attention_type}`

For example, if you want to run experiments for IMDB dataset with Tanh attention, please use `python train_and_run_experiments_bc.py --dataset imdb --data_dir . --output_dir outputs/ --attention tanh`

