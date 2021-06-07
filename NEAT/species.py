from operator import attrgetter
import math
import random
from random import gauss
import numpy as np

#### Species Config
# Variables
excess_coeff = 1.5
large_gene_normalizer = 200
weight_diff_coeff = 1
compatibility_threshold = 3
# This variable is used in analyzing if the new best fitness is
# greater than the current best fitness + fitness_comparison
fitness_comparison = 0.0

# Config = {
#     excess_coeff = 1.5
#     lgn = 1000
#     weight_diff_coeff = 1
#     compatibility_threshold = 3
# }


def setConfig(config):
    global excess_coeff, large_gene_normalizer, weight_diff_coeff, compatibility_threshold
    excess_coeff = config["excess_coeff"]
    large_gene_normalizer = config["lgn"]
    weight_diff_coeff = config["weight_diff_coeff"]
    compatibility_threshold = config["compatibility_threshold"]


def get_information(self, brain_1, brain_2):
    matching = 0
    total_diff = 0.0
    for gene_1 in brain_1.genes:
        for gene_2 in brain_2.genes:
            if gene_1.innovation_number == gene_2.innovation_number:
                matching += 1
                total_diff += abs(gene_1.weight - gene_2.weight)
                break
    excess_and_disjoint = brain_1.genes.length + brain_2.genes.length - 2 * matching
    avg_weight_diff = 100.0
    if matching != 0:
        avg_weight_diff = total_diff / matching
    return excess_and_disjoint, avg_weight_diff


class Species:
    def __init__(self, p=None):
        self.players = [] if p is None else [p]
        self.best_fitness = 0 if p is None else p.fitness
        self.champion = None if p is None else p.clone()
        self.average_fitness = 0
        self.std_fitness = 0
        self.staleness = 0

    def same_species(self, genome):
        information = get_information(self.champion.genome, genome)

        compatibility = (excess_coeff * information[0]) / (large_gene_normalizer + weight_diff_coeff * information[1])
        return compatibility < compatibility_threshold

    def add_to_species(self, player):
        self.players.append(player)

    def sort_players(self):
        self.players = sorted(self.players, key=attrgetter('fitness'), reverse=True)

        if len(self.players) == 0:
            self.staleness = 200
            return self.staleness

        if self.players[0].fitness > (self.best_fitness + fitness_comparison):
            self.staleness = 0
            self.best_fitness = self.players[0].fitness
            self.champion = self.players[0].clone()
        else:
            self.staleness += 1

    def set_average(self):
        fitness = []
        for player in self.players:
            fitness.append(player.fitness)
        fitness = np.array(fitness)
        self.average_fitness = np.mean(fitness)
        self.std_fitness = np.std(fitness)

    def select_player(self):
        random_fitness = gauss(self.average_fitness, self.std_fitness)
        s = 0
        for player in self.players:
            s += player.fitness
            if s > random_fitness:
                return player
        return self.players[0]

    def get_child(self, innovation_history):
        baby = None
        r = random.random()
        # 5%
        if len(self.players) < 2 or r < 0.05:
            baby = self.select_player().clone()
        else:
            p1 = self.select_player()
            p2 = self.select_player()
            baby = p1.crossover(p2) if p1.fitness > p2.fitness else p2.crossover(p1)
        baby.genome.mutate(innovation_history)
        return baby

    def cull(self):
        self.players = self.players[0, math.ceil(len(self.players)/2.0)]

    def fitness_sharing(self):
        for player in self.players:
            player.fitness /= len(self.players)
