import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        # Define the encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, action_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, encoded_obs, action):
        combined = torch.cat((encoded_obs, action), dim=1)
        return self.layers(combined)

class InverseDynamicsModel(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size):
        super(InverseDynamicsModel, self).__init__()
        self.encoder = Encoder(observation_size, hidden_size)
        self.action_predictor = nn.Linear(hidden_size * 2, action_size)

    def forward(self, obs1, obs2):
        encoded_obs1 = self.encoder(obs1)
        encoded_obs2 = self.encoder(obs2)
        combined = torch.cat((encoded_obs1, encoded_obs2), dim=1)
        return self.action_predictor(combined)



class ForwardPredictionModel(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size):
        super(ForwardPredictionModel, self).__init__()
        self.encoder = Encoder(observation_size, hidden_size)
        self.decoder = Decoder(action_size, hidden_size, observation_size)

    def forward(self, obs, action):
        encoded_obs = self.encoder(obs)
        predicted_next_obs = self.decoder(encoded_obs, action)
        return predicted_next_obs
