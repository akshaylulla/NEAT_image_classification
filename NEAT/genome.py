import json
from typing import List

import numpy as np

from NEAT.node import Node, nodesCantConnect
from NEAT.connection import Connection
from NEAT.innovation import InnovationHistory
import random

class Genome:
    def __init__(self, inputs, outputs, empty=False):
        # Connections
        self.genes = []
        self.nodes = []
        self.inputs = inputs
        self.outputs = outputs
        self.layers = 2
        self.nextNode = 0
        self.network = []
        self.biasNode = 0
        self.layer_count = []

        if not empty:
            for i in range(inputs):
                # Append a node with id i, and layer 0 to nodes array
                self.nodes.append(Node(i, 0))
                # Keep track of what our next node will be labeled
                self.nextNode += 1

            for i in range(inputs, inputs+outputs):
                # Append a node with id i and layer 1 to nodes array
                self.nodes.append(Node(i, 1))
                # Keep track of nextNode
                self.nextNode += 1

            # Add bias
            self.biasNode = Node(self.nextNode)
            self.nodes.append(self.biasNode)
            self.nextNode += 1
            self.generateNetwork()

    def toJSONObj(self):
        json_data = {'inputs': self.inputs, 'outputs': self.outputs, 'layers': self.layers, 'nextNode': self.nextNode,
                     'biasNode': self.biasNode.id}

        genes = []
        for gene in self.genes:
            genes.append(gene.toJSONObj())
        json_data['genes'] = genes

        nodes = []
        for node in self.nodes:
            nodes.append(node.toJSONObj())
        json_data['nodes'] = nodes

        return json_data

    def loadFromData(self, data):
        self.inputs = data['inputs']
        self.outputs = data['outputs']
        self.layers = data['layers']
        self.nextNode = data['nextNode']

        for d in data['nodes']:
            self.nodes.append(Node(d['id'], d['layer']))

        self.biasNode = self.getNode(data['biasNode'])

        for d in data['genes']:
            to_node = self.getNode(d['to_node'])
            from_node = self.getNode(d['from_node'])
            gene = Connection(from_node, to_node, d['weight'], d['innovation_number'], d['enabled'])
            self.genes.append(gene)

        self.connectNodes()
        self.generateNetwork()

    def getNode(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        return None

    def connectNodes(self):
        for node in self.nodes:
            node.output_connections = []

        for c in self.genes:
            c.from_node.output_connections.append(c)

    def generateNetwork(self):
        self.connectNodes()
        self.network = []

        self.layer_count = []
        for layer in range(self.layers):
            layer_id = 0
            self.layer_count.append(0)
            for node in self.nodes:
                if node.layer == layer:
                    node.layer_id = layer_id
                    layer_id += 1
                    self.layer_count[layer] += 1
                    self.network.append(node)

    def feedForward(self, input_values):
        if not len(input_values) == self.inputs:
            raise Exception("Amount of inputs do not match the amount of input nodes in genome")

        for i, input in enumerate(input_values):
            self.nodes[i].output_sum = input
        self.biasNode.output_sum = 1

        for node in self.network:
            node.engage()

        output_values = []
        for i in range(self.inputs, self.inputs+self.outputs):
            output = self.nodes[i].output_sum
            output_values.append(output)

        for node in self.nodes:
            node.input_sum = 0

        return output_values

    def feedForward2(self, input_values):
        # 1xL1
        X = np.array(input_values)
        X = np.pad(X, (0, 1), 'constant')
        # L1xL2 then L2xL3 then L3xL4 etc until LNx2
        WLayers = []
        for i in range(self.layers-1):
            WLayers.append(np.zeros((self.layer_count[i], self.layer_count[i+1])))
        for gene in self.genes:
            layer = gene.from_node.layer
            WLayers[layer][gene.from_node.layer_id][gene.to_node.layer_id] = gene.weight

        for i in range(self.layers-1):
            X = np.dot(X, WLayers[i])
            # X = 1 / (1 + np.exp(X))
            X = np.tanh(X)

        return X

    def innovateAndConnect(self, innovation_history_array, from_node, to_node, weight):
        new_connection = True
        innovation_number = len(innovation_history_array)

        for innovation in innovation_history_array:
            if innovation.matches(self, from_node, to_node):
                new_connection = False
                innovation_number = innovation.innovation_number

        if new_connection:
            innovation_numbers = []
            for gene in self.genes:
                innovation_numbers.append(gene.innovation_number)

            innovation_history = InnovationHistory(from_node.id, to_node.id, innovation_number, innovation_numbers)

            innovation_history_array.append(innovation_history)

        self.genes.append(Connection(from_node, to_node, weight, innovation_number))

    def isFullyConnected(self):
        max_connections = 0
        nodes_in_layers = [0] * self.layers

        for node in self.nodes:
            nodes_in_layers[node.layer] += 1

        for i in range(self.layers):
            nodesInFront = 0
            for j in range(i+1, self.layers):
                nodesInFront += nodes_in_layers[j]
            max_connections += nodes_in_layers[i] * nodesInFront

        return max_connections == len(self.genes)

    def getAllNodesNotLayer(self, layer):
        return_nodes = []
        for node in self.nodes:
            if node.layer != layer:
                return_nodes.append(node)
        return return_nodes

    def addConnection(self, innovation_history_array):
        if self.isFullyConnected():
            return

        node1 = self.getNode(random.randint(0, len(self.nodes)-1))
        memory_layer = node1.layer
        other_nodes = self.getAllNodesNotLayer(node1.layer)
        node2 = other_nodes[random.randint(0, len(other_nodes)-1)]

        while nodesCantConnect(node1, node2):
            node1 = self.getNode(random.randint(0, len(self.nodes) - 1))
            if node1.layer != memory_layer:
                memory_layer = node1.layer
                other_nodes = self.getAllNodesNotLayer(node1.layer)
            node2 = other_nodes[random.randint(0, len(other_nodes) - 1)]

        if node2.layer < node1.layer:
            tmp = node1
            node1 = node2
            node2 = tmp

        self.innovateAndConnect(innovation_history_array, node1, node2, random.uniform(-1, 1))

        self.connectNodes()

    def addNode(self, innovation_history_array):
        if len(self.genes) < 1:
            self.addConnection(innovation_history_array)
            return

        gene = self.genes[random.randint(0, len(self.genes)-1)]
        gene.enabled = False

        new_node = Node(self.nextNode, gene.from_node.layer + 1)
        self.nextNode += 1

        to_node = gene.to_node

        self.innovateAndConnect(innovation_history_array, gene.from_node, new_node, 1)
        self.innovateAndConnect(innovation_history_array, new_node, to_node, gene.weight)

        if not gene.from_node.id == self.biasNode.id:
            self.innovateAndConnect(innovation_history_array, self.biasNode, new_node, 0)

        if new_node.layer == to_node.layer:
            for node in self.nodes:
                if node.layer >= new_node.layer:
                    node.layer += 1
            self.layers += 1

        self.nodes.append(new_node)

        self.generateNetwork()

    def mutate(self, innovation_history):
        if len(self.genes) < 1:
            self.addConnection(innovation_history)

        r = random.random()

        # 80%
        if r < 0.8:
            for gene in self.genes:
                gene.mutateWeight()
        # 15%
        elif r < 0.95:
            self.addConnection(innovation_history)
        # 5%
        else:
            self.addNode(innovation_history)

    def matchingGene(self, parent2, innovation_number):
        index = -1
        for i, gene2 in enumerate(parent2.genes):
            if gene2.innovation_number == innovation_number:
                index = i
        return index

    def crossover(self, parent2):
        child = Genome(self.inputs, self.outputs, True)
        child.layers = self.layers
        child.nextNode = self.nextNode
        child.biasNode = self.biasNode
        child_genes = []
        enabled_list = []

        for gene_1 in self.genes:
            gene_2_index = self.matchingGene(parent2, gene_1.innovation_number)
            if gene_2_index >= 0:
                gene_2 = parent2.genes[gene_2_index]
                if not (gene_2.enabled and gene_1.enabled):
                    r = random.random()
                    # 25%
                    if r < 0.25:
                        enabled_list.append(True)
                    else:
                        enabled_list.append(False)
                else:
                    enabled_list.append(True)
                r = random.random()
                # Chance for gene to push from parent 2 or parent 1
                # Default 50%
                if r < 0.5:
                    child_genes.append(gene_1)
                else:
                    child_genes.append(gene_2)
            else:
                child_genes.append(gene_1)
                enabled_list.append(gene_1.enabled)

        for node in self.nodes:
            child.nodes.append(node.clone())

        for index, gene in enumerate(child_genes):
            gene_clone = gene.clone(child.getNode(gene.from_node.id), child.getNode(gene.to_node.id))
            gene_clone.enabled = enabled_list[index]
            child.genes.append(gene_clone)

        child.generateNetwork()
        return child

    def clone(self):
        new_genome = Genome(self.inputs, self.outputs, True)

        for node in self.nodes:
            new_genome.nodes.append(node.clone())

        for gene in self.genes:
            gene_clone = gene.clone(new_genome.getNode(gene.from_node.id), (new_genome.getNode(gene.to_node.id)))
            new_genome.genes.append(gene_clone)

        new_genome.layers = self.layers
        new_genome.nextNode = self.nextNode
        new_genome.biasNode = self.biasNode
        new_genome.generateNetwork()

        return new_genome
