import time
import sys
from datetime import datetime

import uuid

from os.path import dirname, abspath, join
from os import mkdir

from Player import Player

p = dirname(abspath(__file__))

print(p)

p += "\\trials\\" + str(datetime.now()).replace(':', '-')

print(p)

mkdir(p)

player = Player(10, 2)

ih = []
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)
player.genome.mutate(ih)

player.save(p + '\\' + 'Gen-' + str(27) + '.json')