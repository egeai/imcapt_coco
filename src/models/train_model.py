import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        """Load the pretrained ResNet-152 and replace top fully connected layer"""
        super(CNNModel, self).__init__()
        resnet = models.resnet152(pretrained=True)
        module_list = list(resnet.children())[:-1]  # delete the last fully-connected layer
        self.resnet_module = nn.Sequential(*module_list)
        self.linear_layer = nn.Linear(resnet.fc.in_features, embedding_size)
        self.batch_norm = nn.BatchNorm1d(embedding_size, momentum=0.01)

    def forward(self, input_images):
        """Extract feature vectors from input images.
        When feature extracting, we only want to update the parameters of the last layer,
        or in other words, we only want to update the parameters for the layer(s) we are reshaping.
        Therefore, we do not need to compute the gradients of the parameters that we are not changing
        """
        with torch.no_grad():
            resnet_features = self.resnet_module(input_images)
        resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
        final_features = self.batch_norm(self.linear_layer(resnet_features))
        return final_features


class LSTMModel(nn.Module):
    def __init__(self, embedding_size, hidden_layer_size, vocabulary_size, num_layers, max_seq_len=20):
        """Set the hyper parameters and build the layers."""
        super(LSTMModel, self).__init__()
        self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm_layer = nn.LSTM(embedding_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear_layer = nn.Linear(hidden_layer_size, vocabulary_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embedding_layer(captions)
        # torch.cat, concatenates the given sequence of seq tensors in the given dimension
        embeddings = torch.cat((input_features.unsqueeze(1), embeddings), 1)
        lstm_input = pack_padded_sequence(input=embeddings, lengths=lengths, batch_first=True)
        hidden_variables, _ = self.linear_layer(lstm_input)
        model_outputs = self.linear_layer(hidden_variables[0])
        return model_outputs

    def sample(self, input_features, lstm_states=None):
        """Generate captions for given image feature using greedy search."""
        sampled_indices = []
        lstm_inputs = input_features.unsqueeze(1)
        for i in range(self.max_seq_len):
            hidden_variables, lstm_states = self.lstm_layer(lstm_inputs,
                                                            lstm_states)  # hidden(s): (batch_size, 1, hidden_size)
            model_outputs = self.linear_layer(hidden_variables.squeeze(1))  # outputs: (batch_size, vocab_size)
            _, predicted_outputs = model_outputs.max(1)                     # predicted: (batch_size)
            sampled_indices.append(predicted_outputs)
            lstm_inputs = self.embedding_layer(predicted_outputs)  # inputs: (batch_size, embed_size)
            lstm_inputs = lstm_inputs.unsqueeze(1)                 # inputs: (batch_size, 1, embed_size)
        sampled_indices = torch.stack(sampled_indices, 1)          # sampled_ids: (batch_size, max_seq_length)
        return sampled_indices

