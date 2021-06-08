from NEAT.genome import Genome
import numpy as np
import json


def getSimpleScore(actual_value, classification):
    # 1 = cat, 0 = dog
    cat_or_dog = -1
    if classification[0] > classification[1]:
        cat_or_dog = 1
    else:
        cat_or_dog = 0

    if actual_value == cat_or_dog:
        return 1
    else:
        return 0


def getScore(actual_value, classification):
    pass


class Player:
    def __init__(self, image_height, image_width, empty=False):
        self.image_height = image_height
        self.image_width = image_width
        self.gen = 0
        self.vision = np.array([0] * image_height * image_width, dtype=float)
        self.classification = {"cat": 0, "dog": 0}

        if not empty:
            self.genome = Genome(image_width * image_height, 2)

        self.fitness = 0

    def save(self, filename):
        jsonFile = open(filename, "w")
        jsonFile.write(json.dumps(self.toJSONObj()))
        jsonFile.close()

    def toJSONObj(self):
        return {'image_height': self.image_height, 'image_width': self.image_width, 'gen': self.gen,
                'genome': self.genome.toJSONObj()}

    def load(self, filename):
        file = open(filename, "r")
        json_data = file.read()
        data = json.loads(json_data)
        self.image_height = data['image_height']
        self.image_width = data['image_width']
        self.gen = data['gen']
        self.genome = Genome(None, None, empty=True)
        self.genome.loadFromData(data['genome'])

    def test(self, data):
        total = 0
        score = 0
        for d in data:
            image_data = d[0]
            c = d[1]

            self.classification = self.genome.feedForward(image_data)

            score += getSimpleScore(c, self.classification)
            total += 1

        self.fitness = score / total
        if self.fitness == 1:
            print("WHAT THE FUCK")

    def clone(self):
        new_player = Player(self.image_height, self.image_width)
        new_player.genome = self.genome.clone()
        return new_player

    def crossover(self, parent_2):
        baby_genome = self.genome.crossover(parent_2.genome)
        baby = Player(0, 0, empty=True)
        baby.genome = baby_genome
        return baby

    def __str__(self):
        output = "Gen: " + str(self.gen) + " Layers: " + str(self.genome.layers) + "\nConnections: "
        for gene in self.genome.genes:
            output += str(gene) + " // "
        output += '\n'
        for node in self.genome.nodes:
            output += str(node) + " // "
        output += '\nFitness: ' + str(self.fitness) + '\n'
        return output
