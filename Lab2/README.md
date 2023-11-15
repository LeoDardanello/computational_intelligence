# Nim-Second Assignment

# Rules of the game

Given a number of rows, for each row start the game with an incresing number of odd objects (i.e: 1,3,5,7...). Players alternates each turn to remove a choosen amount of object from a single row. The player that remove the last of object loses.

# Goal of the assignment:

Implement an Evolutionary Strategy to compute a strategy able to play against an player that always perform the best possible move

# Approach implemented:

Defined two custom set of rules that represents the possible moves of the player at each turn.
The first set of rules is made of 4 custom moves, some of the rules have some random elements while other don't.

The second set of rules is made of 3 moves,two custom and the optimal move, and its used for debuggin the ES

The sets of moves are use to Train an (μ+λ) Evolutionary Strategy that uses a distinct self-adapting variance (σ) for every rule defined in the strategy.
The choice of a "plus strategy" was preferred to avoid losing the best solution found so far 