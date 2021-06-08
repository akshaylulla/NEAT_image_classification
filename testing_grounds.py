import random
import time
import sys
from datetime import datetime

import uuid

from os.path import dirname, abspath, join
from os import mkdir

from NEAT.genome import Genome
from Player import Player

genome = Genome(1000, 2)

ih = []
inputs = []
for i in range(10000):
    genome.mutate(ih)

for i in range(1000):
    inputs.append(random.random())

print(len(ih))
print("Traditional Test: ")
start = time.time()
print(genome.feedForward(inputs))
end = time.time()
print("Time elapsed: ", end-start)
print("Matrix test: ")
start = time.time()
print(genome.feedForward2(inputs))
end = time.time()
print("Time elapsed: ", end-start)