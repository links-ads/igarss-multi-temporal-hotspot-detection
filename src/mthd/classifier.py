import torch


class BinaryClassifier(torch.nn.module):

    def __init__(self, num_features):
        super(BinaryClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(num_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x
