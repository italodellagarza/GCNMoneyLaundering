import torch
from torch.nn import init, Parameter, LeakyReLU, Module, Softmax, Linear
from torch import cat, DoubleTensor


class NodeLevelAttentionLayer(Module):
    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        super(NodeLevelAttentionLayer, self).__init__()
        self.weight_node = Parameter(DoubleTensor(node_feature_size, node_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        self.weight_edge = Parameter(DoubleTensor(edge_feature_size, edge_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        self.parameter_vector_node = Parameter(DoubleTensor(2*node_embed_size))
        self.parameter_vector_edge = Parameter(DoubleTensor(node_embed_size + edge_embed_size))
        self.importance_normalization = Softmax(dim=0)
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()

    def forward(self, node_features, edge_features, adjacency_matrix, edge_adjacency_matrix):
        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)
        outputs = []
        for i in range(node_features.shape[0]):
            # Selecionar os vizinhos
            features_neighbors_n = node_embeds[adjacency_matrix[:,i] > 0]
            concat_result_n = torch.cat((node_embeds[i].expand(features_neighbors_n.shape[0], features_neighbors_n.shape[1]), features_neighbors_n), 1)
            attention_output_n = self.node_activation(torch.matmul(self.parameter_vector_node.reshape([1,-1]), concat_result_n.T)[0])
            importance_coeficients_n = self.importance_normalization(attention_output_n)
            # Coeficientes de importancia obtidos
            neighbors_n = node_features[adjacency_matrix[:,i] > 0]
            means_n = torch.mean(importance_coeficients_n.reshape([-1,1]).mul(neighbors_n), 0)
            output_nodes = self.node_activation(
                torch.matmul(
                    means_n,
                    self.weight_node
                )
            )
            # Encontrar x_E para toda aresta k vizinha do nó
            features_neighbors_e = edge_embeds[edge_adjacency_matrix[:,i] > 0]
            concat_result_e = torch.cat((node_embeds[i].expand(features_neighbors_e.shape[0], node_embeds.shape[1]), features_neighbors_e), 1)
            attention_output_e = self.edge_activation(torch.matmul(self.parameter_vector_edge.reshape([1,-1]), concat_result_e.T)[0])
            
            importance_coeficients_e = self.importance_normalization(attention_output_e)
            # Coeficientes de importancia obtidos
            neighbors_e = edge_features[edge_adjacency_matrix[:,i] > 0]
            means_e = torch.mean(importance_coeficients_e.reshape([-1,1]).mul(neighbors_e), 0)
            output_edges = self.edge_activation(
                torch.matmul(
                    means_e,
                    self.weight_edge
                )
            )
            outputs.append(torch.cat([output_nodes, output_edges]))
            # Concatenar x_N e x_E
        return torch.stack(outputs)


class EdgeLevelAttentionLayer(Module):
    def __init__(self, node_feature_size, edge_feature_size, node_embed_size, edge_embed_size):
        super(EdgeLevelAttentionLayer, self).__init__()
        self.weight_node = Parameter(DoubleTensor(node_feature_size, node_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_node, gain=1.0)
        self.weight_edge = Parameter(DoubleTensor(edge_feature_size, edge_embed_size))
        torch.nn.init.xavier_uniform_(self.weight_edge, gain=1.0)
        self.parameter_vector_edge = Parameter(DoubleTensor(2*edge_embed_size))
        self.parameter_vector_node = Parameter(DoubleTensor(edge_embed_size + node_embed_size))
        self.importance_normalization = Softmax(dim=0)
        self.node_activation = LeakyReLU()
        self.edge_activation = LeakyReLU()


    # TODO renomear matrizes de adjacencia
    def forward(self, node_features, edge_features, edge_to_edge_adj_matrix, node_to_edge_adj_matrix):
        node_embeds = torch.matmul(node_features, self.weight_node)
        edge_embeds = torch.matmul(edge_features, self.weight_edge)
        outputs = []
        for i in range(edge_features.shape[0]):

            features_neighbors_n = node_embeds[node_to_edge_adj_matrix[:,i] > 0]
            concat_result_n = torch.cat((edge_embeds[i].expand(features_neighbors_n.shape[0], edge_embeds.shape[1]), features_neighbors_n), 1)
            attention_output_n = self.node_activation(torch.matmul(self.parameter_vector_node.reshape([1,-1]), concat_result_n.T)[0])
            importance_coeficients_n = self.importance_normalization(attention_output_n.T)
            # Coeficientes de importancia obtidos
            neighbors_n = node_features[node_to_edge_adj_matrix[:,i] > 0]
            means_n = torch.mean(importance_coeficients_n.reshape([-1,1]).mul(neighbors_n), 0)
            output_nodes = self.node_activation(
                torch.matmul(
                    means_n,
                    self.weight_node
                )
            )
            features_neighbors_e = edge_embeds[edge_to_edge_adj_matrix[:,i] > 0]
            concat_result_e = torch.cat((edge_embeds[i].expand(features_neighbors_e.shape[0], features_neighbors_e.shape[1]), features_neighbors_e), 1)
            attention_output_e = self.edge_activation(torch.matmul(self.parameter_vector_edge.reshape([1,-1]), concat_result_e.T)[0])
            importance_coeficients_e = self.importance_normalization(attention_output_e)
            # Coeficientes de importancia obtidos
            neighbors_e = edge_features[edge_to_edge_adj_matrix[:,i] > 0]
            means_e = torch.mean(importance_coeficients_e.reshape([-1,1]).mul(neighbors_e), 0)
            output_edges = self.edge_activation(
                torch.matmul(
                    means_e,
                    self.weight_edge
                )
            )
            outputs.append(torch.cat([output_nodes, output_edges]))
        return torch.stack(outputs)



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
        print(edge_embeds1)
        node_embeds = self.layer2(node_features, edge_embeds1, node_to_node_adj_matrix, edge_to_node_adj_matrix)
        #print(node_embeds)
        edge_embeds2 = self.layer3(node_embeds, edge_embeds1, edge_to_edge_adj_matrix, node_to_edge_adj_matrix)
        output_linear = self.layer4(edge_embeds2.float())
        output = self.layer5(output_linear)
        return output
