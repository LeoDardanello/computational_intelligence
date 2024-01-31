import random
from gameTraining import  Move
from gameTraining import GameTraining
from MyPlayer import DQLPlayer
from game import Game
import os

class RandomPlayer():
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        return from_pos, move


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




