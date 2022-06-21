import torch
from torch.nn import init, Parameter, LeakyReLU, Module, Softmax, Linear
from torch import cat, FloatTensor


class NodeLevelAttentionLayer(Module):
    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        super(NodeLevelAttentionLayer, self).__init__()
        self.weight_node = Parameter(FloatTensor(node_feature_size, node_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        self.weight_edge = Parameter(FloatTensor(edge_feature_size, edge_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        self.parameter_vector_node = Parameter(FloatTensor(2*node_embed_size))
        self.parameter_vector_edge = Parameter(FloatTensor(node_embed_size + edge_embed_size))
        self.importance_normalization = Softmax(dim=0)
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()
        self.edge_embed_size = edge_embed_size
        self.node_embed_size = node_embed_size

    def forward(self, node_features, edge_features, node_to_node_adj_matrix, edge_to_node_adj_matrix):

        num_nodes = node_features.shape[0]
        num_edges = edge_features.shape[0]

        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)
        diagonal_n = torch.diag_embed(node_to_node_adj_matrix.T)
        diagonal_e = torch.diag_embed(edge_to_node_adj_matrix.T)
        features_neighbors_n = torch.matmul(diagonal_n, node_embeds)
        features_neighbors_e = torch.matmul(diagonal_e, edge_embeds)

        concat_result_n = torch.cat(
            (
                node_embeds.repeat_interleave(num_nodes, dim=0).reshape([num_nodes, num_nodes, self.node_embed_size]),
                features_neighbors_n
            ),
            2
        )

        attention_output_n = self.node_activation(torch.matmul(concat_result_n, self.parameter_vector_node))
        importance_coeficients_n = self.importance_normalization(attention_output_n)
        means_n = torch.mean(importance_coeficients_n.reshape([num_nodes,num_nodes,1]) * torch.matmul(diagonal_n, node_features), 1)

        output_nodes = self.node_activation(
            torch.matmul(
                means_n,
                self.weight_node
            )
        )


        concat_result_e = torch.cat(
            (
                node_embeds.repeat_interleave(num_edges, dim=0).reshape(num_nodes, num_edges, self.node_embed_size),
                features_neighbors_e
            ),
            2
        )
        attention_output_e = self.edge_activation(torch.matmul(concat_result_e, self.parameter_vector_edge))
        importance_coeficients_e = self.importance_normalization(attention_output_e)
        means_e = torch.mean(importance_coeficients_e.reshape([num_nodes, num_edges, 1]) * torch.matmul(diagonal_e, edge_features), 1)

        output_edges = self.edge_activation(
            torch.matmul(
                means_e,
                self.weight_edge
            )
        )
        return torch.cat((output_nodes, output_edges), dim=1)


class EdgeLevelAttentionLayer(Module):
    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        super(EdgeLevelAttentionLayer, self).__init__()
        self.weight_node = Parameter(FloatTensor(node_feature_size, node_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        self.weight_edge = Parameter(FloatTensor(edge_feature_size, edge_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        self.parameter_vector_edge = Parameter(FloatTensor(2*edge_embed_size))
        self.parameter_vector_node = Parameter(FloatTensor(edge_embed_size + node_embed_size))
        self.importance_normalization = Softmax(dim=1)
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()
        self.edge_embed_size = edge_embed_size
        self.node_embed_size = node_embed_size


    # TODO renomear matrizes de adjacencia
    def forward(self, node_features, edge_features, edge_to_edge_adj_matrix, node_to_edge_adj_matrix):

        num_nodes = node_features.shape[0]
        num_edges = edge_features.shape[0]


        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)
        diagonal_n = torch.diag_embed(node_to_edge_adj_matrix.T)
        diagonal_e = torch.diag_embed(edge_to_edge_adj_matrix.T)
        features_neighbors_n = torch.matmul(diagonal_n, node_embeds)
        features_neighbors_e = torch.matmul(diagonal_e, edge_embeds)

        concat_result_n = torch.cat(
            (
                edge_embeds.repeat_interleave(num_nodes, dim=0).reshape([num_edges, num_nodes, self.edge_embed_size]),
                features_neighbors_n
            ),
            2
        )
        attention_output_n = self.node_activation(torch.matmul(concat_result_n, self.parameter_vector_node))
        importance_coeficients_n = self.importance_normalization(attention_output_n)
        means_n = torch.mean(importance_coeficients_n.reshape([num_edges,num_nodes,1]) * torch.matmul(diagonal_n, node_features), 1)

        output_nodes = self.node_activation(
            torch.matmul(
                means_n,
                self.weight_node
            )
        )

        concat_result_e = torch.cat(
            (
                edge_embeds.repeat_interleave(num_edges, dim=0).reshape(num_edges, num_edges, self.edge_embed_size),
                features_neighbors_e
            ),
            2
        )
        attention_output_e = self.edge_activation(torch.matmul(concat_result_e, self.parameter_vector_edge))
        importance_coeficients_e = self.importance_normalization(attention_output_e)
        means_e = torch.mean(importance_coeficients_e.reshape([num_edges, num_edges, 1]) * torch.matmul(diagonal_e, edge_features), 1)

        output_edges = self.edge_activation(
            torch.matmul(
                means_e,
                self.weight_edge
            )
        )
        return torch.cat((output_nodes, output_edges), dim=1)



class Nenn(Module):
    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size, class_size):
        # TODO Seguir estratégia descrita no artigo Nó->Aresta->Nó
        super(Nenn, self).__init__()
        intermediate_size = node_embed_size + edge_embed_size
        self.layer1 = EdgeLevelAttentionLayer(node_feature_size, edge_feature_size, node_embed_size, edge_embed_size)
        self.layer2 = NodeLevelAttentionLayer(node_feature_size, intermediate_size, node_embed_size, edge_embed_size)
        self.layer3 = EdgeLevelAttentionLayer(intermediate_size, intermediate_size, node_embed_size, edge_embed_size)
        self.layer4 = Linear(intermediate_size, class_size)
        self.layer5 = Softmax(dim=1)

    def forward(
        self,
        node_features,
        edge_features,
        edge_to_edge_adj_matrix,
        edge_to_node_adj_matrix,
        node_to_edge_adj_matrix,
        node_to_node_adj_matrix,
    ):
        edge_embeds1 = self.layer1(node_features, edge_features, edge_to_edge_adj_matrix, node_to_edge_adj_matrix)
        node_embeds = self.layer2(node_features, edge_embeds1, node_to_node_adj_matrix, edge_to_node_adj_matrix)
        #print(node_embeds)
        edge_embeds2 = self.layer3(node_embeds, edge_embeds1, edge_to_edge_adj_matrix, node_to_edge_adj_matrix)
        output_linear = self.layer4(edge_embeds2.float())
        output = self.layer5(output_linear)
        return output, edge_embeds2
