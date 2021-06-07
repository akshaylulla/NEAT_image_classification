import math


# Type of sigmoid
# Options = "tanh", "sigmoid"
type = "tanh"

def setConfig(config):
    type = config["type"]

def sigmoid(val):
    if type == "tanh":
        return math.tanh(val)
    elif type == "sigmoid":
        return 1 / (1 + math.exp(-val))
    else:
        return val

def nodesCantConnect(node1, node2):
    return node1.isConnectedTo(node2) or node1.layer == node2.layer

class Node:
    def __init__(self, id, layer=0):
        self.id = id
        self.input_sum = 0.0
        self.output_sum = 0.0
        self.output_connections = []
        self.layer = layer

    def engage(self):
        if not self.layer == 0:
            self.output_sum = sigmoid(self.input_sum)
        for c in self.output_connections:
            if c.enabled:
                c.to_node.input_sum += self.output_sum * c.weight

    def isConnectedTo(self, node):
        if self.layer == node.layer:
            return False

        first_node = self
        second_node = node
        if first_node.layer > second_node.layer:
            first_node = node
            second_node = self

        for c in first_node.output_connections:
            if c.to_node is second_node:
                return True
        return False

    def clone(self):
        return Node(self.id, self.layer)

    def __str__(self):
        return str(self.id)