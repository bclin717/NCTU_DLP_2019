import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DecoderRNN(nn.Module):
    def __init__(
            self, word_size, hidden_size, latent_size, condition_size
    ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_size = word_size

        self.latent_to_hidden = nn.Linear(
            latent_size + condition_size, hidden_size
        )
        self.word_embedding = nn.Embedding(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, word_size)

    def initHidden(self, z, c):
        latent = torch.cat((z, c), dim=2)
        return self.latent_to_hidden(latent)

    def forward(self, x, hidden):
        # get (1, 1, hidden_size)
        x = self.word_embedding(x).view(1, 1, self.hidden_size)

        # get (1, 1, hidden_size) (1, 1, hidden_size)
        output, hidden = self.gru(x, hidden)

        # get (1, word_size)
        output = self.out(output).view(-1, self.word_size)

        return output, hidden

    def forwardv1(self, inputs, z, c, teacher=False, hidden=None):
        # get (1,1,latent_size + condition_size)
        latent = torch.cat((z, c), dim=2)

        # get (1,1,hidden_size)
        if hidden is None:
            hidden = self.latent_to_hidden(latent)

        # get (seq, 1, hidden_size)
        x = self.word_embedding(inputs).view(-1, 1, self.hidden_size)

        input_length = x.size(0)

        # get (seq, 1, hidden_size), (1, 1, hidden_size)
        if teacher:
            outputs = []
            for i in range(input_length - 1):
                output, hidden = self.gru(x[i:i + 1], hidden)
                hidden = x[i + 1:i + 2]
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)
        else:
            # Omit EOS token
            x = x[:-1]
            outputs, hidden = self.gru(x, hidden)

        # get (seq, word_size)
        outputs = self.out(outputs).view(-1, self.word_size)

        return outputs, hidden
