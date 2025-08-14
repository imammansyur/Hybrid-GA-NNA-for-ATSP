# Hybrid-GA-NNA-for-ATSP
This is a MATLAB code for hybrid Genetic Algorithm (GA) with Nearest Neighbor Algorithm (NNA) for ATSP problems. Its supposed to be used with TSPLIB dataset and SPBU Surabaya dataset.

# How To Use
- Set whether to use TSPLIB dataset or SPBU Surabaya dataset in the beginning of the code.
- Adjust the parameters if needed

# Input
You can set whether to use TSPLIB dataset or SPBU Surabaya dataset in the beginning of the code.
## TSPLIB
This takes anything in the folder of `ALL_ATSP` with the format `.atsp`. 
## SPBU Surabaya
It needs `distance-matrix.csv` of NxN data and `places.csv` that contains info of each points including the coordinates. The coordinates then used to make a visualization.

# Output
In the `outputs` folder, each run of the program is grouped in a folder with the format of '[Timestamps]-[GA/GA-NNA]'. Each folder contains the results of each run of every dataset in csv and the visualization in png.

# Known Issue
Its made with the purpose of running it with either TSPLIB dataset or SPBU Surabaya dataset. Therefore, the code will need some modifications to fit another dataset.
