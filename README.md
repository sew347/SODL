# Spectral_Subspace_DL
Dictionary learning using spectral methods.

## Basic workflow (simulated data)

### Creating a simulated dictionary matrix and sample data (`dict_sample.py`):
```bash 
python dict_sample.py --M 100 --s 4 --K 200 --N 4000 --seed 1 --output_folder simulated_data
```

### Recovering spanning subspaces (`subspace_recovery.py`):
```bash 
python subspace_recovery.py --sample_file simulated_data/Y.npy --s 4 --output_folder simulated_data/recov_subspaces
```

### Recovering dictionary from subspaces (`subspace_intersection.py`):
```bash 
python subspace_intersection.py --subspace_folder simulated_data/recov_subspaces --output_folder simulated_data/results --tau 0.5 --J 1000 --eta 0.5
```

