from torch import stack, tril
from torch.nn import Module


from torch.nn import Parameter, Linear
from torch import tensor, tensordot, ones, matmul, zeros
from torch.nn.functional import relu, softmax
from torch import movedim


class PerceptronModel(Module):
    def __init__(self, dimensions):
        super(PerceptronModel, self).__init__()

        "*** YOUR CODE HERE ***"
        self.w = Parameter(ones(1, dimensions))


    def get_weights(self):
        return self.w

    def forward(self, x):
        "*** YOUR CODE HERE ***"
        return matmul(x, self.w.T).reshape(1)


    def get_prediction(self, x):
        score = self(x)

        "*** YOUR CODE HERE ***"
        return 1 if self(x).item() >= 0 else -1


class RegressionModel(Module):

    def __init__(self):

        "*** YOUR CODE HERE ***"
        super().__init__()
        self.layer1 = Linear(1, 128)
        self.layer2 = Linear(128, 128)
        self.layer3 = Linear(128, 1)


    def forward(self, x):
        "*** YOUR CODE HERE ***"
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)


class DigitClassificationModel(Module):

    def __init__(self):

        super().__init__()
        input_size = 28 * 28
        output_size = 10
        "*** YOUR CODE HERE ***"
        self.fc1 = Linear(input_size, 256)
        self.fc2 = Linear(256, 128)
        self.fc3 = Linear(128, output_size)

    def forward(self, x):
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return self.fc3(x)


class LanguageIDModel(Module):

    def __init__(self):


        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        super(LanguageIDModel, self).__init__()
        "*** YOUR CODE HERE ***"

        self.hidden_size = 192

        self.input_layer = Linear(self.num_chars, self.hidden_size)
        self.recurrent_layer = Linear(self.hidden_size, self.hidden_size)

        self.classifier_hidden = Linear(self.hidden_size, 96)
        self.classifier_output = Linear(96, len(self.languages))


    def forward(self, xs):
        "*** YOUR CODE HERE ***"
        hidden = relu(self.input_layer(xs[0]))

        for character in xs[1:]:
            hidden = relu(
                self.input_layer(character)
                + self.recurrent_layer(hidden)
            )

        hidden = relu(self.classifier_hidden(hidden))

        return self.classifier_output(hidden)


def Convolve(input: tensor, weight: tensor):
    input_tensor_dimensions = input.shape
    weight_dimensions = weight.shape
    Output_Tensor = tensor(())
    "*** YOUR CODE HERE ***"
    output_height = input_tensor_dimensions[0] - weight_dimensions[0] + 1
    output_width = input_tensor_dimensions[1] - weight_dimensions[1] + 1

    Output_Tensor = zeros((output_height, output_width))

    for y in range(weight_dimensions[0]):
        for x in range(weight_dimensions[1]):
            region = input[
                y:y + output_height,
                x:x + output_width
            ]

            Output_Tensor = Output_Tensor + region * weight[y, x]
    return Output_Tensor


class DigitConvolutionalModel(Module):

    def __init__(self):

        super().__init__()
        output_size = 10

        self.convolution_weights = Parameter(ones((3, 3)))
        """ YOUR CODE HERE """
        self.fc1 = Linear(26 * 26, 128)
        self.fc2 = Linear(128, output_size)

    def forward(self, x):
        x = x.reshape(len(x), 28, 28)
        x = stack(
            list(map(lambda sample: Convolve(sample, self.convolution_weights), x))
        )
        x = x.flatten(start_dim=1)
        """ YOUR CODE HERE """
        x = relu(self.fc1(x))
        return self.fc2(x)

class Attention(Module):
    def __init__(self, layer_size, block_size):
        super().__init__()
        self.k_layer = Linear(layer_size, layer_size)
        self.q_layer = Linear(layer_size, layer_size)
        self.v_layer = Linear(layer_size, layer_size)


        self.register_buffer(
            "mask",
            tril(ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

        self.layer_size = layer_size

    def forward(self, input):
        B, T, C = input.size()

        """YOUR CODE HERE"""


        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)


        scores = matmul(Q, movedim(K, 1, 2)) / (self.layer_size ** 0.5)


        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, float('-inf')
        )[0]


        weights = softmax(scores, dim=-1)


        return matmul(weights, V)

