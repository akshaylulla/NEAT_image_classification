import math
from operator import attrgetter
from Player import Player
from NEAT.species import Species
import sys

max_staleness = 20

class Population:
    def __init__(self, size, number_of_size=1):
        self.innovation_history = []
        self.best_player = None
        self.best_players_by_gen = []
        self.best_fitness = 0
        self.gen = 0
        self.species = []
        self.kill_everyone_LOL = False
        self.pop_life = 0
        self.done = False
        self.population = []
        if number_of_size < 1:
            number_of_size = 1
        for i in range(number_of_size):
            tmp_player = Player(100, 100)
            tmp_player.genome.mutate(self.innovation_history)
            self.population.append(tmp_player)

        self.toolbar_width = 100
        self.player_increment = len(self.population) // 50

    def naturalSelection(self):
        print('Running natural selection for generation: ' + str(self.gen))
        self.speciate()
        self.calculatePlayersFitness()
        self.sortSpecies()
        if self.kill_everyone_LOL:
            # this.species.splice(massExtinctionConfig, this.species.length - massExtinctionConfig);
            #self.species = self.species[0:]
            # https://www.geeksforgeeks.org/remove-multiple-elements-from-a-list-in-python/
            del self.species[massExtinctionConfig:(len(self.species)-massExtinctionConfig)]
            self.kill_everyone_LOL = False
        self.cullSpecies()
        self.setBestPlayer()
        self.killStaleAndBadSpecies()
        avgSum = self.getAvgFitnessSum()
        newGeneration = []
        debug = (self.gen % 10) == 0
        if debug:
            print('========================================================================')
            print('Generation: ', self.gen + 1, ' // Number of Mutations: ', self.innovationHistory.length,
                        ' // Number of Species: ', self.species.length)
            print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>')
        for index, specimen in enumerate(self.species):
            if debug:
                print('#', index + 1, ' Species: ' + specimen.name + ' Best Fitness (UA): ',
                            specimen.bestFitness, ' -----------------')
                print('(', specimen, ')')
            for playerRank,player in enumerate(specimen.players):
                if debug and (index < 4 or playerRank > len(specimen) - 6):
                    print('=== #', playerRank + 1, ": ", player.name, " -------- Fitness: ", player.fitness, ' // Score: ', player.score, ' ----------------')
            newGeneration.append(specimen.champion.clone())
            noOfChildren = math.floor(specimen.averageFitness / avgSum * len(self.population) * len(self.population[0])) - 1
            for i in range(noOfChildren):
                    newGeneration.append(specimen.getChild(self.innovationHistory, self.gen + 1))

        while newGeneration.length < (len(self.population) * len(self.population[0])):
            newGeneration.append(self.species[0].getChild(self.innovationHistory, self.gen + 1))
        i = 0
        for i in range(len(self.population)):
            #self.population[i] = newGeneration.slice(i * len(self.population[i]), (i+1) * len(self.population[i]))
            del newGeneration[i* len(self.population[i]) : (i+1) * len(self.population[i]) ]
            self.population[i] = newGeneration
        self.gen += 1
        self.pop_life = 0
        self.done = False
        #this.snakeID = Math.floor(Math.random() * this.population[this.currentPool].lengt


    def simulatePopulation(self, data):
        print('Running population simulation for generation: ' + str(self.gen))
        # setup toolbar
        sys.stdout.write("[%s]" % (" " * self.toolbar_width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.toolbar_width + 1))  # return to start of line, after '['
        counter = 0
        for player in self.population:
            counter += 1
            if counter > self.player_increment:
                counter = 0
                sys.stdout.write("-")
                sys.stdout.flush()
            player.test(data)
        sys.stdout.write("]\n")  # this ends the progress bar

    def setBestPlayer(self):
        tmp_best = self.species[0].players[0]
        tmp_best.gen = self.gen
        self.best_players_by_gen.append(tmp_best.clone())

        if tmp_best.fitness > self.best_fitness:
            print("=-=-=-=-=-=-=-=-=-=-=-=\nNew King:\n", str(tmp_best))
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

    def killStaleAndBadSpecies(self):
        if len(self.species) < 2:
            return

        for i in range(2, len(self.species)):
            if self.species[i].staleness >= max_staleness:
                del self.species[i]
                i -= 0

        avgSum = self.getAvgFitnessSum()
        for i in range(1, len(self.species)):
            if self.species[i].average_fitness / avgSum * len(self.population) < 1:
                del self.species[i]
                i -= 0

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


