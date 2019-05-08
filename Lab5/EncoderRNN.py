import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(
            self, word_size, hidden_size, latent_size,
            num_condition, condition_size
    ):
        super(EncoderRNN, self).__init__()
        self.word_size = word_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.latent_size = latent_size

        self.condition_embedding = nn.Embedding(num_condition, condition_size)
        self.word_embedding = nn.Embedding(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, inputs, init_hidden, input_condition):
        c = self.condition(input_condition)

        # get (1,1,hidden_size)
        hidden = torch.cat((init_hidden, c), dim=2)

        # get (seq, 1, hidden_size)
        x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)

        # get (seq, 1, hidden_size), (1, 1, hidden_size)
        outputs, hidden = self.gru(x, hidden)

        # get (1, 1, hidden_size)
        m = self.mean(hidden)
        logvar = self.logvar(hidden)

        z = self.sample_z() * torch.exp(logvar / 2) + m

        return z, m, logvar

    def initHidden(self):
        return torch.zeros(
            1, 1, self.hidden_size - self.condition_size,
            device=device
        )

    def condition(self, c):
        c = torch.LongTensor([c]).to(device)
        return self.condition_embedding(c).view(1, 1, -1)

    def sample_z(self):
        return torch.normal(
            torch.FloatTensor([0] * self.latent_size),
            torch.FloatTensor([1] * self.latent_size)
        ).to(device)
