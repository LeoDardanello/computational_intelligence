# HalloweenChallenge

Solution developed with [PaolaMts](https://github.com/PaolaMts)

Our solution uses a single state hill climber algorithm to solve the set covering problem.

It combines two differents optimizations : 
- Steepest Step
- Simulated Annealing

The tweak function chooses the best successor sorting the sets with respect to the overlapping and the number of false (not covered) in each set

The fitness function takes into consideration the number of True (covered) to estimate the cost (quality) of the solution.

The cooling factor for the Simulated Annealing comes from "Monte Carlo Statistical Methods" by Christian P. Robert and George Casella
