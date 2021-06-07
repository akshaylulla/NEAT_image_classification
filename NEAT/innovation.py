class InnovationHistory:
    def __init__(self, from_id, to_id, innovation_number, innovation_numbers_array):
        self.from_id = from_id
        self.to_id = to_id
        self.innovation_number = innovation_number
        self.innovation_array = innovation_numbers_array

    def matches(self, genome, from_node, to_node):
        if len(genome.genes) == len(self.innovation_array) \
                and from_node.id == self.from_id \
                and to_node.id == self.to_id:
            for gene in genome.genes:
                if not gene.innovation_number in self.innovation_array:
                    return False
            return True
        return False
