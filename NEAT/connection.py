from random import gauss
import random

#### Weight Mutation
# Chance for large number mutation (1-lnm is the chance for gaussian change)
lnm = 0.15
# Large Number Range
lnrange_config = (-1.1, 1.1)
# Gaussian Mu, Sigma
gauss_config = (0, 1)
# Standardize weights between x,y
standardize = True
standardize_range = (-1, 1)

# Config = {
#   lnm: 0.15,
#   lnrange: (-1.1, 1.1),
#   gauss_mu: 0,
#   gauss_sigma: 1,
#   standardize: True,
#   standardize_range: (-1, 1)
# }
def setConfig(config):
    global lnm, lnrange_config, gauss_config, standardize, standardize_range
    lnm = config["lnm"]
    lnrange_config = config["lnrange"]
    gauss_config = (config["gauss_mu"], config["gauss_sigma"])
    standardize = config["standardize"]
    standardize_range = config["standardize_range"]


class Connection:
    def __init__(self, from_node, to_node, weight, innovation_number, enabled=True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def mutateWeight(self):
        r = random.random()
        if r < lnm:
            self.weight += random.uniform(lnrange_config[0], lnrange_config[1])
        else:
            self.weight += gauss(gauss_config[0], gauss_config[1])

        if standardize:
            if self.weight < standardize_range[0]:
                self.weight = standardize_range[0]
            elif self.weight > standardize_range[1]:
                self.weight = standardize_range[1]

    def clone(self, from_node, to_node):
        if from_node is None or to_node is None:
            raise Exception("From or To Node not specified when cloning Connection")
        return Connection(from_node, to_node, self.weight, self.innovation_number, self.enabled)

    def __str__(self):
        return str(self.from_node) + " to " + str(self.to_node) + ": " + str(self.weight)

    def toJSONObj(self):
        json_data = {'to_node': self.to_node.id, 'from_node': self.from_node.id, 'weight': self.weight,
                     'enabled': self.enabled, 'innovation_number': self.innovation_number}
        return json_data

