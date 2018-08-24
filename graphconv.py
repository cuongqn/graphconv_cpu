import torch
import torch.nn as nn


class GraphConv(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(GraphConv, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(input_features, output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.initialize_params()

    def initialize_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def graph_pool(self, input, adj):
        output = torch.zeros(input.size(), dtype=torch.float)
        for idx, row in enumerate(adj):
            pooled,_ = torch.max(input[row.nonzero()].squeeze(),dim=0,keepdim=True)
            output[idx] = pooled
        return output

    def forward(self, input, adj):
        intermediate = torch.mm(input, self.weight)
        output = torch.mm(adj, intermediate)
        if self.bias is not None:
            output = output + self.bias
        output = output.clamp(min=0)
        output = self.graph_pool(output, adj)
        return output


class GraphGather(nn.Module):
    def __init__(self):
        super(GraphGather, self).__init__()

    def forward(self, input):
        output = torch.sum(input,dim=0)
        return output


class GraphConvNN(nn.Module):
    def __init__(self, n_conv, n_fc, input_dim, output_dim, conv_dims, fc_dims):
        assert n_conv == len(conv_dims) and n_fc == len(fc_dims), 'Number of dimensions provided does not match number of layers'
        super(GraphConvNN, self).__init__()
        self.n_conv = n_conv
        self.n_fc = n_fc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv_dims = conv_dims
        self.fc_dims = fc_dims
        self.add_conv_layers()
        self.add_fc_layers()

    def add_conv_layers(self):
        for i in range(self.n_conv):
            if i == 0: dim0 = self.input_dim
            else: dim0 = self.conv_dims[i-1]
            dim1 = self.conv_dims[i]
            setattr(self, 'conv_layer_%s' % (i+1), GraphConv(dim0, dim1))
        setattr(self, 'graph_gather', GraphGather())

    def add_fc_layers(self):
        for i in range(self.n_fc):
            if i == 0: dim0 = self.conv_dims[-1]
            else: dim0 = self.fc_dims[i-1]
            dim1 = self.fc_dims[i]
            fc_layer = nn.Sequential(
                nn.Linear(dim0, dim1),
                nn.ReLU()
            )
            setattr(self, 'fc_layer_%s' % (i+1), fc_layer)
        setattr(self, 'output_layer', nn.Linear(self.fc_dims[-1], self.output_dim))

    def forward(self, input, adj):
        assert input.shape[1] == self.input_dim, 'Provided input_dim (%s) does not match node feature length (%s)' % (self.input_dim, input.shape[1])
        input = torch.tensor(input, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)

        descriptors = self.conv_layer_1(input, adj)
        for i in range(1, self.n_conv):
            descriptors = getattr(self, 'conv_layer_%s' % (i+1))(descriptors, adj)
        descriptors_gathered = self.graph_gather(descriptors)

        fc = self.fc_layer_1(descriptors_gathered)
        for i in range(1, self.n_fc):
            fc = getattr(self, 'fc_layer_%s' % (i+1))(fc)

        output = self.output_layer(fc)
        return output
