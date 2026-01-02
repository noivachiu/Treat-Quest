## Treat Quest - Reinforcement Learning Game

Programmer: Noiva Chiu

# Program Description

This program runs a treat hunting simulation, in which a virtual mouse is trained to find all the treats and exit on a generated environment. The mouse is trained using a reinforcement learning algorithm, specifically through Q-Learning. The mouse agent is rewarded +100 points for finding a treat and punished with -1 points for every step or -25 for hitting a trap.

After the agent completes its training phase, the mouse agent is able to take the shortest route to find each treat and makes it way home while avoid all traps.

# File Purposes

* main
  *  managaes main control flow
* env
  * defines environment states and methods
* agent
  * previous learning agent, does not use q-learning
* q_agent
  * current learning agent, defines agent states and behaviours which use q-learning
 
Running Instructions
1. Open a terminal and navigate to the directory containing the program's files
2. Enter this command to execute the project: python main.py





