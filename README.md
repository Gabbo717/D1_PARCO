# D1_PARCO
Repository made to share my work with the professors of Parallel Computing 
I will be dividing this file in 3 unique sections: one for each operative system you might use, I will begin with the windows environment which is the one I have worked with.

1° Section, Windows environment:
Before starting you have to make sure to have installed: a working version of wsl (windows subsystem for linux) or any other software that allows you to operate using linux inside of windows (this depends on your preference, just remember that I have both python and c++ codes to compile so there will be differences in the compilation process if you're yousing, say, a VM to operate and in that case I suggest following the Linux's environment section), python and gcc, you will need them to compile and run my programs (at the latest version possible) and a VPN to connect to the cluster (in case you're working from home).
1) To start: open 3 terminals (you'll need all of them later) and use the cd command to reach the destination you plan of using to download the content of my repository (you could use mkdir to create a new folder but I will suggest using a git command that already creates a folder with a custom name so I do not recommend it), do this for two terminals;
2) Use one of the terminals to clone the repository using the command "git clone https://github.com/Gabbo717/D1_PARCO.git gabriele_bazzanella_bauer_235266" (the directory name contains my full name and my student ID, if you want to change it to something elde you're free to do so), a folder named gabriele_bazzanella_bauer_235266 should have appeared in the environment;
3) Use the command "cd gabriele_bazzanella_bauer_235266" to enter the directory containing the files present in my repository, you should do this for the two terminals; 
4) Use the command "scp *.pbs .\MatrixTranspExpl_Final.cpp .\MatrixTranspImp_Final.cpp .\MatrixTranspSeq_Final.cpp .\PerformanceImprovement_Data.cpp YourUsername@hpc.unitn.it:/home/YourUsername/gabriele_bazzanella_bauer_235266" to push only the necessary programs to the cluster (the .pbs files combined with the .cpp files for data collection), you will likely have to input your university password;
5) Now, a new directory named gabriele_bazzanella_bauer_235266 should have appeared in your home directory in your cluster node, to access it, use the third terminal I made you open in point 1 and remained unused since (if you're from home, make sure you're connected to vpn-mfa.icts.unitn.it through your VPN before proceeding) and input the command "ssh YourUsername@hpc.unitn.it", you will likely have to input your password again, then you will enter the cluster, use the command "ls" to see if the new directory appeared, enter that directory using the command "cd gabriele_bazzanella_bauer_235266";
6) This is very important, before submitting the jobs, you have to modify all of the .pbs files by using the "vim NameFile.pbs", after opening the vim interaction window, press the "i" key to go into INSERT mode, switch the cd command in the file by changing the "cd /home/Your_Username/gabriele_bazzanella_bauer_235266" line with the replacement of your cluster's username with "Your_Username" in the file path, this will make sure to have the output files to be outputted in the gabriele_bazzanella_bauer_235266 directory, unfortunately, I cannot cover for this step myself as I do not know the exact username you will use;
7) to run the jobs, you will have to use the command "qsub NameFile.pbs" for all of the 4 .pbs files present in the directory, each of them take a different amount of time to finish up, the longest one should not take more than 10 minutes;
8) Once every job has been completed, check for any errors in each of the 4 .err files using the command "cat ErrorFileName.err", if all of them are empty (which they should be) then you are ready to export the output files;
9) Switch to one of the other two terminals (which should both be positioned in the gabriele_bazzanella_bauer_235266 directory inside of your local machine) and use the command "scp YourUsername@hpc.unitn.it:/home/YourUsername/gabriele_bazzanella_bauer_235266/*.out ./" to import all of the .out files at once in your local machine (you, once again, probably will have to input your password);
10) Once all of the .out files have been downloaded from the cluster, you can freely clear out all of the files and the directory in your cluster, access that one terminal still open in your cluster and call these three commands in order: "rm *" to delete all of the files inside of the directory, "cd .." to exit the directory and "rm -r gabriele_bazzanella_bauer_235266" to remove the empty directory;
11) 
