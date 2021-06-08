import json
import math
import os.path
from datetime import datetime
from operator import attrgetter
from Player import Player
from NEAT.species import Species
import sys
import uuid
from os.path import dirname, abspath
from os import mkdir
from joblib import Parallel, delayed

max_staleness = 20

acc_mutation = 1

species_data = []
avg_fitness_data = []
innovation_data = []

class Population:
    def __init__(self, number_of_size, IMG_size):
        path = dirname(dirname(abspath(__file__)))
        path += "\\trials\\" + str(datetime.now()).replace(':', '-')
        mkdir(path)
        self.path = path
        print('Saving best models to: ' + path)

        self.toolbar_width = 50
        self.player_increment = number_of_size // self.toolbar_width

        self.pop_folder = path
        self.innovation_history = []
        self.best_player = None
        self.best_players_by_gen = []
        self.best_fitness = 0
        self.gen = 0
        self.species = []
        self.pop_life = 0
        self.done = False
        self.population = []

        print('Generating Population...')
        sys.stdout.write("[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        counter = 0
        for i in range(number_of_size):
            counter += 1
            if counter > self.player_increment:
                counter = 0
                sys.stdout.write("-")
                sys.stdout.flush()
            tmp_player = Player(IMG_size, IMG_size)
            for i in range(acc_mutation):
                tmp_player.genome.mutate(self.innovation_history)
            self.population.append(tmp_player)
        sys.stdout.write("]\n")  # this ends the progress bar

    def naturalSelection(self):
        print('Running natural selection for generation: ' + str(self.gen))
        # print('Starting speciation and culling...')
        self.speciate()
        self.sortSpecies()
        self.cullSpecies()
        self.setBestPlayer()
        self.killStaleAndBadSpecies()

        # print('Finished. Generating new generation...')
        self.gen += 1
        new_generation = []
        avgSum = self.getAvgFitnessSum()
        sys.stdout.write("[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        counter = 0

        for species in self.species:
            champion = species.champion.clone()
            champion.gen = self.gen
            new_generation.append(champion)
            counter += 1
            no_children = (species.average_fitness // avgSum * len(self.population)) - 1
            i = 0
            while i < no_children:
                counter += 1
                if counter > self.player_increment:
                    counter = 0
                    sys.stdout.write("-")
                    sys.stdout.flush()
                new_generation.append(species.get_child(self.innovation_history, self.gen))
                i += 1

        while len(new_generation) < len(self.population):
            counter += 1
            if counter > self.player_increment:
                counter = 0
                sys.stdout.write("-")
                sys.stdout.flush()
            new_generation.append(self.species[0].get_child(self.innovation_history, self.gen))

        sys.stdout.write("]\n")  # this ends the progress bar
        avg_fitness_data.append(avgSum)
        innovation_data.append(len(innovation_data))
        self.saveData()
        self.population = new_generation

    def saveData(self):
        filename = self.path + '\\data.json'
        jsonFile = open(filename, "w")
        data = {'species_data': species_data,
                'avg_fitness_data': avg_fitness_data,
                'innovation_data': innovation_data}
        json_data = json.dumps(data)
        jsonFile.write(json_data)
        jsonFile.close()

    def simulatePopulation(self, data):
        print('Running population simulation for generation: ' + str(self.gen))
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        """counter = 0
        for player in self.population:
            counter += 1
            if counter > self.player_increment:
                counter = 0
                sys.stdout.write("-")
                sys.stdout.flush()
            print("before - ", player.fitness)
            player.test(data)
            print("after - ", player.fitness)"""

        def run_together(player):
            player.test(data)

        # Parallel(n_jobs=-1, verbose=10)(delayed(run_together)(player) for player in self.population) - with verbose
        Parallel(n_jobs=-1, backend="threading")(delayed(run_together)(player) for player in self.population)
        sys.stdout.write("]\n")  # this ends the progress bar

    def setBestPlayer(self):
        tmp_best = self.species[0].players[0]
        tmp_best.gen = self.gen
        self.best_players_by_gen.append(tmp_best.clone())

        filename = "Gen-" + str(self.gen) + "-F" + str(self.best_fitness) + ".json"
        tmp_best.save(self.pop_folder + '\\' + filename)
        print(tmp_best.fitness)
        print(self.best_fitness)
        if tmp_best.fitness > self.best_fitness:
            self.best_fitness = tmp_best.fitness
            print("=-=-=-=-=-=-=-=-=-=-=-=\nNew King:\n", str(tmp_best))
            print("Saving data to file: " + filename)
            print('=-=-=-=-=-=-=-=-=-=-=-=')

    def speciate(self):
        for species in self.species:
            species.players = []

        for player in self.population:
            species_found = False
            for species in self.species:
                if species.same_species(player.genome):
                    species.add_to_species(player)
                    species_found = True
                    break
            if not species_found:
                self.species.append(Species(player))

    def sortSpecies(self):
        for species in self.species:
            species.sort_players()
        self.species = sorted(self.species, key=attrgetter('best_fitness'), reverse=True)

        gen_species_data = {}
        for species in self.species:
            gen_species_data[str(species.uuid)] = {'amt_nn': len(species.players)}
            species_data.append(gen_species_data)

    def killStaleAndBadSpecies(self):
        if len(self.species) < 2:
            return

        i = 2
        while i < len(self.species):
            if self.species[i].staleness >= max_staleness:
                del self.species[i]
                i -= 0
            i += 1

        avgSum = self.getAvgFitnessSum()
        i = 1
        while i < len(self.species):
            if self.species[i].average_fitness / avgSum * len(self.population) < 1:
                del self.species[i]
                i -= 0
            i += 1

    def cullSpecies(self):
        for species in self.species:
            species.cull()
            species.fitness_sharing()
            species.set_average()

    def getAvgFitnessSum(self):
        avgSum = 0
        for species in self.species:
            avgSum += species.average_fitness
        return avgSum


