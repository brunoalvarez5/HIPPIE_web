This is the source code for the HIPPIE web. HIPPIE is a deep learning framework that classifies neurons from extracelular recordings, these recordings include waveforms and spiking dynamics. 
The web is developed so that through a user friendly inteface users can submit data to ve processed HIPPIE and visualized.

It accepts 2 types of inputs, .csv files and .acqm files. For the .csv files it needs the acg (autocorrelograms), isi distribution and waveforms data in 3 different files, this files need to not have a header.
In the case of the acqm files it only needs one containing all the date since it will later on be digested by a neurocurator to extract the needed information.
The web will take care of everything and display the raw data followed by a parametric UMAP with the clusters computed. If the user already has its own cell types, a .csv file with those can be submited.
After the PUMAP there is a section to visualize how data differes between clusters and the user can select which cluster to highlight.
Also all the computed data and files can be downloaded in case the user wants to perform or make a deeper research.

The code essentially ceners around the web_code.py file. This file executes and runs all the functions needed and some basic computations. The Neurocurator.py is a python script that digests the data from the 
.acqm file so that it can be used. Then all the other functions are stored in the utils.py file.
