README
===============================================================================

Markov Decision Processes  

-------------------------------------------------------------------------------

Overview:
---------
This repository contains the implementation for **Assignment 4** of **CS-7641 (Spring 2025)**.  
The main script, `main.py`, implements and analyzes Markov Decision Processes (MDPs) across different environments. It includes:

- **Dynamic Programming methods**: Value Iteration (VI) and Policy Iteration (PI)
- **Reinforcement Learning algorithms**: SARSA and Q-Learning
- **Environment analysis**: Blackjack (discrete, stochastic) and CartPole (continuous, deterministic)
- **Exploration strategies**: Epsilon-greedy, Boltzmann, and UCB
- **Discretization analysis**: Impact of state space granularity on performance and computation

Note: Instructions are written for Linux but apply equally to Windows with minor changes (e.g., virtual environment activation syntax).

-------------------------------------------------------------------------------

Directory Structure:
--------------------
The repository has the following layout:

- `README.txt` ............. This file with setup and usage instructions  
- `main.py` ................ Primary script implementing all experiments and analysis  
- `requirements.txt` ....... List of Python package dependencies 
- `nliddawi3-analysis.pdf` . Analysis report (~8 pages) containing results and insights
- `results/` ............... Auto-generated directory containing output files, organized into:
    - `blackjack/` ......... Results from Blackjack environment
    - `cartpole/` .......... Results from CartPole environment
    - `hyperparameter_tuning/` . Parameter optimization analysis

-------------------------------------------------------------------------------

Dependencies and Setup:
----------------------
Python version: **3.8 or higher** (Python 3.10 recommended)

Required packages:

- `numpy` ............ Numerical computing  
- `matplotlib` ....... Plotting  
- `seaborn` .......... Statistical visualization  
- `pandas` ........... Data manipulation
- `tqdm` ............. Progress bars
- `gymnasium` ........ Reinforcement learning environments
- `bettermdptools` ... MDP utilities (installed from GitHub)

**Option 1: Install individually**  
pip install numpy matplotlib seaborn pandas tqdm gymnasium
pip install git+https://github.com/jlm429/bettermdptools.git

**Option 2: Install from requirements.txt**  
pip install -r requirements.txt

-------------------------------------------------------------------------------

Running the Code:
----------------
To execute the code and reproduce the results, follow these detailed steps:

   1. **Clone the Repository:**
      - Open a command prompt (on Windows) or terminal (on Linux).
      - Run:
            git clone https://github.com/NaderLiddawi/Learning-in-Contrast.git

   2. **Set Up the Python Environment:**
      - Create a virtual environment:
            python -m venv env
      - Activate the virtual environment:
            env\Scripts\activate   (on Windows)
         or
            source env/bin/activate   (on Linux)

   3. **Install Dependencies:**
      - Install all required packages:
            pip install -r requirements.txt

   4. **Run the Script:**
      - Execute the main Python script:
            python main.py

   5. **Review Results:**
      - All visualizations and analysis outputs will be saved in the `results/` directory
      - The script will provide progress updates and summaries in the console

Note: The experiments may take some time to complete due to the computational complexity of solving MDPs, especially with finer discretization levels.

-------------------------------------------------------------------------------

Experiment Configuration:
------------------------
You can modify the following parameters in main.py to customize experiments:

- `NUM_SEEDS`: Number of random seeds for robustness (default: 5)
- `RANDOM_SEEDS`: Specific seeds for reproducibility
- `BLACKJACK_EPISODES`: Number of episodes for Blackjack RL training
- `CARTPOLE_EPISODES`: Number of episodes for CartPole RL training
- `CARTPOLE_BINS`: Discretization levels for CartPole state space
- Hyperparameter grids: Values to test during optimization

For shorter execution time, reduce the number of episodes or discretization levels.

-------------------------------------------------------------------------------

Reproducibility and Determinism:
-------------------------------
- All algorithms use fixed random seeds to guarantee reproducibility
- Every experiment captures and logs parameters and results
- Functions are carefully documented for transparency and auditability
- Execution times may vary due to hardware differences, but results should be consistent

-------------------------------------------------------------------------------

Overleaf Project and Final Commit:
--------------------------------
To inspect the final Overleaf report and development notes:

Overleaf (READ ONLY): https://www.overleaf.com/read/vgjtckkmvwmh#035d25


-------------------------------------------------------------------------------

Additional Notes:
---------------
- The implementation follows the assignment requirements of analyzing dynamic programming and reinforcement learning approaches on contrasting MDP environments
- The analysis particularly focuses on comparing VI vs PI convergence rates, discretization effects on CartPole, and exploration strategy impacts on learning performance
- For bugs or reproducibility concerns, check commit history or contact the maintainer

This project adheres to principles of scientific reproducibility and transparent reporting.

===============================================================================