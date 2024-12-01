# D1_PARCO
Repository made to share my work with the professors of Parallel Computing 

This List of actions are proposed for users using the Windows operating system.

Before starting, ensure you have installed: WSL (Windows Subsystem for Linux) or an equivalent Linux environment (e.g., a VM). Choose based on preference, but note differences in Python and C++ compilation may arise. If using a VM, some commands might be different than the ones I give you here, Python and g++ (latest versions, with support for OpenMP and AVX) for program compilation and execution, and a VPN to connect to the cluster, if working remotely.
1) To start: open 3 terminals (you'll need all of them later) and use the cd command to reach the destination you plan of using to download the content of my repository (you could use mkdir to create a new folder but I will suggest using a git command that already creates a folder with a custom name so I do not recommend it), do this for two terminals;
2) Use one of the terminals to clone the repository using the command ```git clone https://github.com/Gabbo717/D1_PARCO.git gabriele_bazzanella_bauer_235266``` (the directory name contains my full name and my student ID, if you want to change it to something elde you're free to do so), a folder named gabriele_bazzanella_bauer_235266 should have appeared in the environment;
3) Use the command ```cd gabriele_bazzanella_bauer_235266``` to enter the directory containing the files present in my repository, you should do this for the two terminals;
4) Now, create a new directory named gabriele_bazzanella_bauer_235266 in your cluster node, to access it, use the third terminal I made you open in point 1 and remained unused since (if you're from home, make sure you're connected to vpn-mfa.icts.unitn.it through your VPN before proceeding) and input the command ```ssh YourUsername@hpc.unitn.it```, you will likely have to input your password again, then you will enter the cluster, use the command ```ls``` to see if the new directory appeared, enter that directory using the command ```cd gabriele_bazzanella_bauer_235266```;
5) Use the command   ```scp *.pbs .\MatrixTranspExpl_Final.cpp .\MatrixTranspImp_Final.cpp .\MatrixTranspSeq_Final.cpp .\PerformanceImprovement_Data.cpp YourUsername@hpc.unitn.it:/home/YourUsername/gabriele_bazzanella_bauer_235266``` to push only the necessary programs to the cluster (the .pbs files combined with the .cpp files for data collection), you will likely have to input your university password, you should then see in your new cluster's directory the new files uploaded;
6) This is very important, before submitting the jobs, you have to modify all of the .pbs files by using the ```vim NameFile.pbs```, after opening the vim interaction window, press the ```i``` key to go into INSERT mode, switch the cd command in the file by changing the ```cd /home/Your_Username/gabriele_bazzanella_bauer_235266``` line with the replacement of your cluster's username with ```Your_Username``` in the file path, this will make sure to have the output files to be outputted in the gabriele_bazzanella_bauer_235266 directory, unfortunately, I cannot cover for this step myself as I do not know the exact username you will use;
7) to run the jobs, you will have to use the command ```qsub NameFile.pbs``` for all of the 4 .pbs files present in the directory, each of them take a different amount of time to finish up, the longest one should not take more than 10 minutes;
8) Once every job has been completed, check for any errors in each of the 4 .err files using the command ```cat ErrorFileName.err```, if all of them are empty (which they should be) then you are ready to export the output files;
9) Switch to one of the other two terminals (which should both be positioned in the gabriele_bazzanella_bauer_235266 directory inside of your local machine) and use the command ```scp YourUsername@hpc.unitn.it:/home/YourUsername/gabriele_bazzanella_bauer_235266/*.out ./``` to import all of the .out files at once in your local machine (you, once again, probably will have to input your password);
10) Once all of the .out files have been downloaded from the cluster, you can freely clear out all of the files and the directory in your cluster, access that one terminal still open in your cluster and call these three commands in order: ```rm *``` to delete all of the files inside of the directory, ```cd ..``` to exit the directory and ```rm -r gabriele_bazzanella_bauer_235266``` to remove the empty directory;
11) There are only a few last steps before seeing the graphs: open one of the two terminals already present in the gabriele_bazzanella_bauer_235266 directory in your local machine and digit the command ```wsl``` to access the linux subsystem (if you're using another method unfortunately I cannot know that, just proceed acccording to what your have installed) and make sure you're still in the correct directory;
12) Compile the ProcessData.cpp program by using the command g++ ProcessData.cpp (I will assume you will not use the -o flag to rename the executable file, so I will refer to it as a.out);
13) The usage of the program is ./a.out <input_file> <process_option>, for each file there is a different processing option precisely fit for a specific input file, the process_option can be one of 4 alternatives: 1 for SequentialDataOutput.out, 2 for ImplicitParDataOutput.out, 3 for ExplicitParDataOutput.out and lastly 4 for PerformanceImprovementDataOutput.out, so in total you're looking to run 4 times the command ./a.out <input_file> <process_option> so be careful with the combinations otherwise errors in the file parsing will arise.
14) Once you've finished with point 13, you should see that 4 new files have appeared, all called ```ProcessedData_Something```, now it's time to plot this data, switch to the terminal not in wsl but in windows (or just use the terminal with wsl and use the ```exit``` command to go back to windows, your choice) and run the python program using the command ```python PlotData.py``` (if you're using a VM or something that bind you into staying in a Linux system, just input the command ```python3 PlotData.py```), you will be asked to then insert the name of the file, and then the plotting option, just put the right combinations based on the name of the files you have to input and the correct plot of the data should appear on your screen.
15) If you want to check the performance of only one run of the program locally, you can do it by following these steps: open wsl again by using the ```wsl``` command, compile the program MatrixTransposeDeliverable.cpp using this exact command: ```g++ -std=c++11 -fopenmp -O2 MatrixTransposeDeliverable.cpp -mavx``` (You might need to download some libraries if you're missing either the OMP or the AVX one) then, you can run the executable file by using the command ```./a.out <matrix_size> <number_of_threads>```
