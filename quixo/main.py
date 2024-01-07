import random
from game import Game, Move, Player
from MyPlayer import DQLPlayer,HumanPlayer


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])

        return from_pos, move


# class MyPlayer(Player):
#     def __init__(self) -> None:
#         super().__init__()

#     def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
#         from_pos = (random.randint(0, 4), random.randint(0, 4))
#         move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
#         return from_pos, move

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

if __name__ == '__main__':
    g = Game()
    # g.print()
    player1 = DQLPlayer(1,0) # 1 epsilon, 0 symbol
    player2 = RandomPlayer()
    res = g.play(player1, player2)
    # print("#########################")
    # g.print()
    # print(f"Winner: Player {winner}")
    print(res)
    print("DQLPlayer vs RandomPlayer winnig rate: ", (res[0]/(res[0]+res[1]))*100,"%")
    print("Model Trained")
    ans=input("Would you like to play against the trained AGENT? (y/n)")
    if ans=="y":
        print("Cells indexs:")
        print_cell_indexs()
        HumanPlayer=HumanPlayer(1)
        res=g.play_testing(player1,HumanPlayer)
        print("results: ", res)

