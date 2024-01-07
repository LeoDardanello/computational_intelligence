import torch

class DQN(torch.nn.Module):
    def __init__(self,learning_rate,gamma=0.5):
        super().__init__()
        """
        States encoding: Top row, right col, bottom row, left col
        Directions encoding: top, right, bottom,  left
        """
        self.model=torch.nn.Sequential(
            torch.nn.Linear(25, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 44) #number of possible states, 12 cells with 3 possible directions each
        )                            #4 cells (corners) with 2 possible directions each
        self.gamma=gamma
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        return self.Xavier_init(self.model)

    def Xavier_init(self,model):
        for layer in model:
            if type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01) # bias is initialized to 0.01 

    def forward(self, x):
        return self.model(x)
    
    