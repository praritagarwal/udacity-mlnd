# Capstone Project - Machine Learning Engineer Nanodegree
My capstone project for Udacity's Machine Learning Nanodegree

The capstone proposal can be found [here](https://github.com/praritagarwal/udacity-mlnd/blob/master/Capstone%20project/Proposal.pdf).

The proposal was reviewed [here](https://review.udacity.com/#!/reviews/1904389).

The capstone report can be found [here](https://github.com/praritagarwal/udacity-mlnd/blob/master/Capstone%20project/Report.pdf).

# Problem Statement: [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling)
To predict the NMR coupling constants given the atomic positions 

The original data can be downloaded from [here](https://www.kaggle.com/c/champs-scalar-coupling/data). However, for demonstration purposes we have provided a subset of the original data in this repository. The original data contains 85003 molecules for training and 45772 molecules for test submissions. Our sample files consist of a subset of 1000 randomly chosen molecules from the training set and 600 randomly chosen molecules from the test set.

Our jupyter notebooks have been configured to run on the sample files we have provided. In order to run them on the original data, it is easiest to download the orginal data from the link provided above and replace our sample files with original data. Also, rename the original files as follows

train.csv --> train_sample.csv

test.csv  --> test_sample.csv

structures.csv --> structures_sample.csv

# Jupyter Notebooks 

- interatomic_distances.ipynb : The structures_sample.csv file contains the (x, y, z) position coordinates of each atom in all the molecules. In this notebook we use this data to obtain distances between all possible atomic pairs in a molecules. This is saved to a new file called  CHAMPS_rel_pos.csv. It takes less than a minute to  execute on our sample data. 

- nearest_neighbors.ipynb : In this notebook, we compute the distances to 3 nearest neighbors of each kind (i.e. C, H, N, O). This data is stored in appropriate files in the subfolder 'sample_data'. This notebook takes a long time to execute. Even on such a small sample data as provided by us, it needs about 15 min for execution. 

- predicting_scalar_coupling_II.ipynb : This notebook contains our main implementation.  It reads the files in 'sample_data' and engineers various features. These are then fed to a variety of neural networks. The training weights of these neural networks are saved in a subfolder called 'SavedModels'. The predictions produced by the trained neural networks are stored in a subfolder called 'predictions_v2'. The final submission for Kaggle are stored in the subfolder called 'sample_submissions'. It takes a little more than 30 min to execute on our sample data. 

