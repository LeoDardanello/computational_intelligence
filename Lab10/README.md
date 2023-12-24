# Lab 10: Tic-Tac-Toe Player with RL

The code implements a Agent trained with Q-Learning that can play tic-tac-toe.
The Agent is trained against a RandomPlayer for a choosen amount of games and than it if possible to play against it in a real game.

The reward function embeds some of the classical and most frequent "board disposition" in order to facilitate training and to assign better rewards.


# Possible improvement

Exploit the simmetries when assigning the reward to compact the code

# Known Bugs

Sometimes when playing against the QPlayer the board won't be printed on the screen, this is most likely a grapichal glitch in the VisualStudio editor, if this happens just insert an already used cell to print and "invalid move" message and force the printing of the board