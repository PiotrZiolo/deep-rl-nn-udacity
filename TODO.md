- Experiment with black box optimization methods for policy search:
    * Steepest ascent hill climbing is a variation of hill climbing that chooses a small number of neighboring policies at each iteration and chooses the best among them.
    * Simulated annealing uses a pre-defined schedule to control how the policy space is explored, and gradually reduces the search radius as we get closer to the optimal solution.
    * Adaptive noise scaling decreases the search radius with each iteration when a new best policy is found, and otherwise increases the search radius.
	
