# HR-prediction-models
In this repository, we can see some of the different significant Python files that I programmed for my thesis. Some of them are not added because are similar procedures with small differences depending on the dataset used. 



First, we have the baseline code where we try the different modes (Raw Temporal and Frequency and Pure Temporal and Frequency) explained in the Thesis for the IEEE SPC Train:

- baseline_IEEE_SPC_Train

  

Secondly, we have the hybrid baseline code for the IEEE SPC Train based on the results obtained in the previous file.

- baselinehybrid_IEEE_SPC_Train



Keep in mind that we have these two files repeated per each dataset with the values that fit better to them.



The next file includes all the functions that data of IEEE SPC Train should go through before is added to the Dataloader: preprocessing functions, data augmentation, and segmentation among others. 

- functions_IEEESPCTrain



This file is also repeated per each dataset.



Then, we have an example of the main function of the Hybrid model using IEEE SPC Train. The mains are equal between them if they use the same dataset:

- main_hybrid_IEEESPCTrain

  

After that, we have the files starting with "models" for each Dataset as some parameters change among them because the sequences are from different sizes. They include the several classes associated with the data-driven models built during my thesis:

- models_BAMI

- models_Dalia

- models_IEEE_SPC_Train

- models_IEEE_SPC_Test

  

Finally, we have an example of a train function for our hybrid model using the IEEE SPC Train dataset which includes part of the function used for the temporal model and for the frequency model.

- traintest_hybrid_IEEESPCTrain



