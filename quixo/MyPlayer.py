import numpy as np
from DQN import DQN
import torch
import copy
from collections import deque
import random
from game import Move


class DQLPlayer():
    def __init__(self,eps=1,batch_size=256,gamma=0.5,moves_before_update=20,learning_rate=0.001,eps_min=0.01,eps_decay=5e-4) -> None:
        self.eps = eps
        self.eps_min = eps_min #if default it will always explore 1% of the time
        self.eps_decay = eps_decay
        self.symbol = None
        self.learning_rate = learning_rate
        self.model=DQN(self.learning_rate,gamma)
        self.model_target=copy.deepcopy(self.model) # create a model to estimate the target (R_t+gamma*max_a(Q(s_t+1,a)))
        self.replay_memory=deque(maxlen=5000)
        self.moves_before_update=moves_before_update
        self.moves_update_counter=moves_before_update
        self.batch_size=batch_size
        self.gamma=gamma
        self.moves_embbeding=[((0,0),Move.RIGHT),((0,0),Move.BOTTOM),
                            ((0,1),Move.RIGHT),((0,1),Move.LEFT),((0,1),Move.BOTTOM),
                            ((0,2),Move.RIGHT),((0,2),Move.LEFT),((0,2),Move.BOTTOM),
                            ((0,3),Move.RIGHT),((0,3),Move.LEFT),((0,3),Move.BOTTOM),
                            ((0,4),Move.LEFT),((0,4),Move.BOTTOM),
                            ((1,0),Move.TOP),((1,0),Move.RIGHT),((1,0),Move.BOTTOM),
                            ((1,4),Move.TOP),((1,4),Move.LEFT),((1,4),Move.BOTTOM),
                            ((2,0),Move.TOP),((2,0),Move.RIGHT),((2,0),Move.BOTTOM),
                            ((2,4),Move.TOP),((2,4),Move.LEFT),((2,4),Move.BOTTOM),
                            ((3,0),Move.TOP),((3,0),Move.RIGHT),((3,0),Move.BOTTOM),
                            ((3,4),Move.TOP),((3,4),Move.LEFT),((3,4),Move.BOTTOM),
                            ((4,0),Move.TOP),((4,0),Move.RIGHT),
                            ((4,1),Move.TOP),((4,1),Move.LEFT),((4,1),Move.RIGHT),
                            ((4,2),Move.TOP),((4,2),Move.LEFT),((4,2),Move.RIGHT),
                            ((4,3),Move.TOP),((4,3),Move.LEFT),((4,3),Move.RIGHT),
                            ((4,4),Move.TOP),((3,0),Move.LEFT)
                            ]

    def set_symbol(self, symbol):
        self.symbol = symbol

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)

    def load_model(self,path):
        self.model.load_state_dict(torch.load(path))

    def check_winner(self,board):
        for i in range(5):
            if all(board[i][j] == board[i][0] and board[i][0] != -1 for j in range(5)):
                return board[i][0] # win on the row i
            if all(board[j][i] == board[0][i] and board[0][i] != -1 for j in range(5)):
                return board[0][i] # win on the col i
        if all(board[i][i] == board[0][0] and board[0][0] != -1 for i in range(5)):
            return board[0][0] # win on the main diagonal
        if all(board[i][4 - i] == board[0][4] and board[0][4] != -1 for i in range(5)):
            return board[0][4] # win on the secondary diagonal
        return None

    def get_ok_moves(self, board):
        row_len, col_len = len(board), len(board[0])

        all_possible_moves_inds=[
                (0, col) for col in range(col_len) # TOP row
            ]+[
                (riga, col_len - 1) for riga in range(1, row_len- 1)# RIGHT col
            ]+[
                (row_len- 1, col) for col in range(col_len - 1, 0, -1) # BOTTOM row
            ]+[
                (riga, 0) for riga in range(row_len- 1, 0, -1)] # LEFT col

        acceptable_moves = []
        for move in all_possible_moves_inds:
            # player can move a blank cell or a cell with his symbol
            if board[move[0]][move[1]] == self.symbol or board[move[0]][move[1]] == -1:
                acceptable_moves.append(move)
        return acceptable_moves

    def check_dir_ok(self,cell,dir):
        if cell[0]==0 and cell[1] in [1,2,3]:   # Top row
            return True if dir in [Move.LEFT,Move.BOTTOM,Move.RIGHT] else False
        if cell[0]==4 and cell[1] in [1,2,3]:   # Bottom row
            return True if dir in [Move.RIGHT,Move.TOP,Move.LEFT] else False
        if cell[1]==0 and cell[0] in [1,2,3]:   # Left col
            return True if dir in [Move.TOP,Move.BOTTOM,Move.RIGHT] else False
        if cell[1]==4 and cell[0] in [1,2,3]:   # Right col
            return True if dir in [Move.TOP,Move.BOTTOM,Move.LEFT] else False
        if cell==(0,0):
            return True if dir in [Move.BOTTOM,Move.RIGHT] else False
        if cell==(0,4):
            return True if dir in [Move.LEFT,Move.BOTTOM] else False
        if cell==(4,0):
            return True if dir in [Move.TOP,Move.RIGHT] else False
        if cell==(4,4):
            return True if dir in [Move.TOP,Move.LEFT] else False
        return False

    def get_ok_dir(self,cell_ind):
        # if a corner is selected
        if cell_ind==(0,0): # top left corner
            return np.random.choice([Move.BOTTOM, Move.RIGHT])
        elif cell_ind==(0, 4): # top RIGHT corner
            return np.random.choice([Move.BOTTOM, Move.LEFT])
        elif cell_ind==(4, 4): # BOTTOM RIGHT corner
            return np.random.choice([Move.TOP, Move.LEFT])
        elif cell_ind==(4, 0): # BOTTOM left corner
            return np.random.choice([Move.TOP, Move.RIGHT])
        
        # if a side is selected
        elif cell_ind[0]==0: # top side
            return np.random.choice([Move.BOTTOM, Move.LEFT, Move.RIGHT])
        elif cell_ind[1]==4: # RIGHT side
            return np.random.choice([Move.TOP, Move.LEFT, Move.BOTTOM])
        elif cell_ind[0]==4: # BOTTOM side
            return np.random.choice([Move.TOP, Move.LEFT, Move.RIGHT])
        elif cell_ind[1]==0: # left side
            return np.random.choice([Move.TOP, Move.RIGHT, Move.BOTTOM])

    def make_move_training(self, board,going_second)->tuple[tuple[int, int], Move]:
        
        if np.random.random() < self.eps:
            possible_moves = self.get_ok_moves(board)
            random_cell_inds=np.random.choice(len(possible_moves)) # random index of the possible move
            random_cell = possible_moves[random_cell_inds] # random tuple of row and col indexs
            random_dir = self.get_ok_dir(random_cell) # random direction 
            self.eps = max(self.eps_min, self.eps-self.eps_decay)
            return  random_cell, random_dir, random_cell_inds
        else:
            if going_second:
                board_tensor=torch.tensor(self.map_board(board.flatten()),dtype=torch.float32) #swap 0 and 1 to make the network 
                                                                                               #see the board from the second player perspective
            else:
                board_tensor=torch.tensor(board.flatten(),dtype=torch.float32)
            action_ind=torch.argmax(self.model.forward(board_tensor)) # forward 
            action=self.moves_embbeding[action_ind] # get the action from the list
            self.eps = max(self.eps_min, self.eps-self.eps_decay)
            return action[0],action[1],action_ind
        

    def continue_DQN_move_training(self,old_board,new_board,action,valid,going_second):
        result=self.check_winner(new_board)
        if result is not None:
            done=True
            if result==self.symbol:
                reward=1 # win
            else:
                reward=-1 # lose
        else:
            done=False #game not finished
            reward=0.1 # small reward for continuing the game
        if not valid:
            reward=-1 # invalid move
    
        if going_second:
            old_board=self.map_board(old_board.flatten())
            new_board=self.map_board(new_board.flatten())
        else:
            old_board=old_board.flatten()
            new_board=new_board.flatten()

        # update the replay memory
        self.replay_memory.append((tuple(old_board),action,reward,tuple(new_board),done))
        # train the model
        if len(self.replay_memory) > self.batch_size:
            minibatch = random.sample(self.replay_memory,self.batch_size) #extract random sampled fromm the replay memory to train the model


            state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*minibatch) #use * and zip to group by columns 
                                                                                                    #(group by state, action, reward, next_state, done)
            q_values = self.model.forward(torch.tensor(state_batch, dtype=torch.float32))
            
            q_values_next_state = self.model_target(torch.tensor(next_state_batch, dtype=torch.float32)) #netowrk to estimate the 
            

            q_values_target = torch.zeros(self.batch_size, 44)
            for i in range(self.batch_size):
                if done_batch[i]:
                    q_values_target[i,action_batch[i]] = reward_batch[i]  # If the state is terminal, the q-value is just the reward
                else:
                                                        #Bellman equation: R_t+gamma*max_a(Q(s_t+1,a))
                    q_values_target[i,action_batch[i]] = reward_batch[i] + self.gamma * torch.max(q_values_next_state[i]) 

            self.model.optimizer.zero_grad()
            loss = self.model.loss_fn(q_values, q_values_target) 
            loss.backward() # backprop
            self.model.optimizer.step() 

            self.moves_update_counter-=1

            if self.moves_update_counter==0:
                self.model_target.load_state_dict(self.model.state_dict())
                self.moves_update_counter=self.moves_before_update
        return

    def make_move(self,game:"Game"):
        board=game.get_board()
        if self.symbol==1:
            board_mod=self.map_board(board.flatten())
            board_tensor=torch.tensor(board_mod,dtype=torch.float32)
        else:
            board_tensor=torch.tensor(board.flatten(),dtype=torch.float32)
        action_ind=torch.argmax(self.model.forward(board_tensor)) # forward
        action=self.moves_embbeding[action_ind]
        # check if move is valid, if is not do a random move to avoid getting stuck
        
        if not((board[action[0]]==-1 or board[action[0]]==self.symbol) and self.check_dir_ok(action[0],action[1])):
            possible_moves = self.get_ok_moves(board)
            random_cell_inds=np.random.choice(len(possible_moves)) # random index of the possible move
            random_cell = possible_moves[random_cell_inds] # random tuple of row and col indexs
            random_dir = self.get_ok_dir(random_cell) # random direction
            return  random_cell, random_dir
        else:
            return action[0],action[1]

    def map_board(self,board):
        new_board = [1 if el == 0 else 0 if el == 1 else el for el in board]
        return new_board



        



