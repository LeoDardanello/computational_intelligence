from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np
from tqdm import tqdm
from game import Move
# Rules on PDF



class GameTraining(object):
    def __init__(self) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player_idx = 1

    def get_board(self):
        '''
        Returns the board
        '''
        return deepcopy(self._board)

    def get_current_player(self) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        for row in self._board:
            formatted_row = [' X ' if cell==1 else ' O ' if cell==0 else ' . ' for cell in row]
            print('|'.join(formatted_row))
            print('-' * (len(row) * 4 - 1))
        return 

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        player = self.get_current_player()
        winner = -1
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return winner is this guy
                winner = self._board[x, 0]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                winner = self._board[0, y]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            winner = self._board[0, 0]
        if winner > -1 and winner != self.get_current_player():
            return winner
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            winner = self._board[0, -1]
        return winner

    def play(self, player1, player2) -> int:
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        n_games= 5000
        results = [0,0]
        
        for i in tqdm(range(n_games)):
            self.current_player_idx = 1 if n_games%2==0 else 0
            going_second = True if n_games%2==0 else False
            winner = -1
            while winner < 0:
                self.current_player_idx += 1
                self.current_player_idx %= len(players)
                ok = False
                while not ok:
                    if players[self.current_player_idx].__class__.__name__=="DQLPlayer":
                       
                        if going_second:
                            players[self.current_player_idx].set_symbol(1)
                        else:
                            players[self.current_player_idx].set_symbol(0) 
                        # DQL agent makes moves 
                        old_board = deepcopy(self._board)
                        from_pos, slide ,action_id= players[self.current_player_idx].make_move_training(self._board,going_second)
                        ok=self.__move(from_pos, slide, self.current_player_idx)
                       
                        valid= True if ok else False
                        new_board = deepcopy(self._board)
                        players[self.current_player_idx].continue_DQN_move_training(old_board,new_board,action_id,valid,going_second)
                    else:
                    
                        # random player keeps trying until it finds a valid move
                        from_pos, slide = players[self.current_player_idx].make_move(self)
                        ok = self.__move(from_pos, slide, self.current_player_idx)
  
                    winner = self.check_winner()
            results[winner]+=1
            print("Game ",i," winner: ",winner," results: ",results)

        return results



    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[0], from_pos[1])])
        acceptable = self.__take((from_pos[0], from_pos[1]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[0], from_pos[1]), slide)
            if not acceptable:
                self._board[(from_pos[0], from_pos[1])] = deepcopy(prev_value)
        return acceptable

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        # print("aceeptable top ",acceptable_top," bottom ",acceptable_bottom," left ",acceptable_left," right ",acceptable_right)
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = self._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                self._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    self._board[(from_pos[0], i)] = self._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    self._board[(i, from_pos[1])] = self._board[(
                        i - 1, from_pos[1])]
                # move the piece up
                self._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    self._board[(i, from_pos[1])] = self._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
        # print("from pos",from_pos," slide ",slide," acceptable ",acceptable)
        return acceptable
