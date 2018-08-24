import pickle
import torch
import torch.nn as nn
from graphconv import GraphConvNN
from graphconv import train_GCNN

def main():
    atom_features = pickle.load(open('data/atom_features.obj', 'rb'))
    adj_mat = pickle.load(open('data/adj_mat.obj', 'rb'))
    y = pickle.load(open('data/logD.obj', 'rb'))

    model = GraphConvNN(n_conv=2,
                        n_fc=2,
                        input_dim=75,
                        output_dim=1,
                        conv_dims=[128, 128],
                        fc_dims=[128, 64])
    criterion = nn.MSELoss(reduction='elementwise_mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train_GCNN(model=model,
                       x=atom_features,
                       adjacency_matrix=adj_mat,
                       y=y,
                       epochs=150,
                       batch_size=128,
                       optimizer=optimizer,
                       criterion=criterion)

if __name__ == '__main__':
    main()
