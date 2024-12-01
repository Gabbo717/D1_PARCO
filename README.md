# D1_PARCO
Repository made to share my work with the professors of Parallel Computing 
I will be dividing this file in 3 unique sections: one for each operative system you might use, I will begin with the windows environment which is the one I have worked with.

1° Section, Windows environment:
Before starting you have to make sure to have installed: a working version of wsl (windows subsystem for linux) or any other software that allows you to operate using linux inside of windows (this depends on your preference, just remember that I have both python and c++ codes to compile so there will be differences in the compilation process if you're yousing, say, a VM to operate and in that case I suggest following the Linux's environment section), python and gcc, you will need them to compile and run my programs (at the latest version possible) and a VPN to connect to the cluster (in case you're working from home).
1) To start: open 3 terminals (you'll need all of them later) and use the cd command to reach the destination you plan of using to download the content of my repository (you could use mkdir to create a new folder but I will suggest using a git command that already creates a folder with a custom name so I do not recommend it), do this for two terminals;
2) Use one of the terminals to clone the repository using the command "git clone https://github.com/Gabbo717/D1_PARCO.git gabriele_bazzanella_bauer_235266" (the directory name contains my full name and my student ID, if you want to change it to something elde you're free to do so), a folder named gabriele_bazzanella_bauer_235266 should have appeared in the environment;
3) Use the command "cd gabriele_bazzanella_bauer_235266" to enter the directory containing the files present in my repository, you should do this for the two terminals; 
4) Use the command "scp .\ExplicitParDataCollection.pbs .\ImplicitParDataCollection.pbs .\SequentialDataCollection.pbs .\MatrixTranspExpl_Final.cpp .\MatrixTranspImp_Final.cpp .\MatrixTranspSeq_Final.cpp YourUsername@hpc.unitn.it:/home/YourUsername/gabriele_bazzanella_bauer_235266" to push only the necessary programs to the cluster (the .pbs files combined with the .cpp files for data collection
