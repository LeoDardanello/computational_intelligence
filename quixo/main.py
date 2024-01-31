import random
from gameTraining import  Move
from gameTraining import GameTraining
from MyPlayer import DQLPlayer,HumanPlayer
from game import Game
import os

class RandomPlayer():
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        return from_pos, move



def print_cell_indexs():
    '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
    board_mapping = [
        [1, ' | ', 2, ' | ', 3, ' | ', 4, ' | ', 5],
        ['-' * 12] * 2,
        [6, ' | ', 7, ' | ', 8, ' | ', 9, ' | ', 10],
        ['-' * 12] * 2,
        [11, ' | ', 12, ' | ', 13, ' | ', 14, ' | ', 15],
        ['-' * 12] * 2,
        [16, ' | ', 17, ' | ', 18, ' | ', 19, ' | ', 20],
        ['-' * 12] * 2,
        [21, ' | ', 22, ' | ', 23, ' | ', 24, ' | ', 25]
    ]

    for row in board_mapping:
        print(''.join(map(lambda x: f"{x:2}", row)))

def play_against_agent():
    print("////////////////////////////////")
    print("////////////////////////////////")
    print("The board is numbered as follows:")
    print("Cells indexs:")
    print_cell_indexs()
    print("You are going second, your symbol is 'X'")
    HumanPlyr=HumanPlayer(1)
    g=GameTraining()
    res=g.play_against_human(player1,HumanPlyr)
    if res==1:
        print("You won!")
    else:
        print("You lost!")

if __name__ == '__main__':
    player2 = RandomPlayer()
    player1 = DQLPlayer()
    if os.path.exists("./network_testing.pth"):
        print("Loading model...")
        player1.load_model("./network_testing.pth")
        print("Model loaded")
    else:    
        # Training the agent
        g = GameTraining()
        print("Training the agent:")
        res = g.play(player1, player2)
        print(res)
        print("Training; DQLPlayer vs RandomPlayer winnig rate: ", (res[0]/(res[0]+res[1]))*100,"%")
        print("Model Trained")
        player1.save_model("./network_testing.pth")
        
    # Testing the agent against a random player
    print("Testing the agent against a random player:")
    g2=Game()
    player1.set_symbol(0)
    win_as_first=0

    for runs in range (10):
        avg=0
        for i in range(100):
            winner=g2.play(player1,player2)
            if winner==0:
                win_as_first+=1
        avg+=win_as_first
    print("Test; wins as first player:",avg/10)

    player1.set_symbol(1)
    win_as_second=0
    for runs in range (10):
        avg=0
        for i in range(100):
            winner=g2.play(player2,player1)
            if winner==1:
                win_as_second+=1
        avg+=win_as_second
    print("Test; wins as second player:",avg/10)
    ans=input("Would you like to play against the trained AGENT? (y/n)")
    if ans=="y":
        play_against_agent()



