import numpy as np
import random
import math

class NodeGene:
    INPUT  = "input"
    HIDDEN = "hidden"
    OUTPUT = "output"

    def __init__(self, node_id, node_type):
        self.node_id   = node_id
        self.node_type = node_type

class ConnectionGene:
    def __init__(self, from_node, to_node, weight, innovation):
        self.from_node  = from_node
        self.to_node    = to_node
        self.weight     = weight
        self.innovation = innovation
        self.enabled    = True

class InnovationTracker:
    def __init__(self):
        self.counter = 0
        self.history = {}

    def get(self, from_node, to_node):
        key = (from_node, to_node)
        if key not in self.history:
            self.history[key] = self.counter
            self.counter += 1
        return self.history[key]

class Genome:
    def __init__(self, input_size, output_size, tracker):
        self.input_size  = input_size
        self.output_size = output_size
        self.tracker     = tracker
        self.nodes       = {}
        self.connections = {}
        self.fitness     = 0

        for i in range(input_size):
            self.nodes[i] = NodeGene(i, NodeGene.INPUT)

        for i in range(output_size):
            nid = input_size + i
            self.nodes[nid] = NodeGene(nid, NodeGene.OUTPUT)

        for i in range(input_size):
            for j in range(output_size):
                to_id  = input_size + j
                innov  = tracker.get(i, to_id)
                weight = np.random.uniform(-1, 1)
                self.connections[innov] = ConnectionGene(i, to_id, weight, innov)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, inputs):
        activations = {}

        for i in range(self.input_size):
            activations[i] = float(inputs[i])

        for _ in range(10):
            for conn in self.connections.values():
                if not conn.enabled:
                    continue
                if conn.from_node not in activations:
                    continue
                prev = activations.get(conn.to_node, 0.0)
                activations[conn.to_node] = prev + activations[conn.from_node] * conn.weight

        outputs = []
        for i in range(self.output_size):
            oid = self.input_size + i
            outputs.append(self.sigmoid(activations.get(oid, 0.0)))
        return outputs

    def mutate_weights(self):
        for conn in self.connections.values():
            if random.random() < 0.8:
                conn.weight += np.random.uniform(-0.5, 0.5)
                conn.weight  = float(np.clip(conn.weight, -3, 3))
            else:
                conn.weight = np.random.uniform(-1, 1)

    def mutate_add_connection(self):
        node_ids = list(self.nodes.keys())
        for _ in range(10):
            a = random.choice(node_ids)
            b = random.choice(node_ids)
            if a == b:
                continue
            if self.nodes[b].node_type == NodeGene.INPUT:
                continue
            if self.nodes[a].node_type == NodeGene.OUTPUT:
                continue
            innov = self.tracker.get(a, b)
            if innov in self.connections:
                continue
            self.connections[innov] = ConnectionGene(a, b, np.random.uniform(-1,1), innov)
            return

    def mutate_add_node(self):
        active = [c for c in self.connections.values() if c.enabled]
        if not active:
            return

        conn         = random.choice(active)
        conn.enabled = False

        new_id = max(self.nodes.keys()) + 1
        self.nodes[new_id] = NodeGene(new_id, NodeGene.HIDDEN)

        i1 = self.tracker.get(conn.from_node, new_id)
        i2 = self.tracker.get(new_id, conn.to_node)

        self.connections[i1] = ConnectionGene(conn.from_node, new_id, 1.0, i1)
        self.connections[i2] = ConnectionGene(new_id, conn.to_node, conn.weight, i2)

    def mutate(self):
        self.mutate_weights()
        if random.random() < 0.05:
            self.mutate_add_node()
        if random.random() < 0.1:
            self.mutate_add_connection()

    def copy(self):
        child = Genome(self.input_size, self.output_size, self.tracker)
        child.nodes       = {k: NodeGene(v.node_id, v.node_type) for k, v in self.nodes.items()}
        child.connections = {}
        for k, v in self.connections.items():
            c         = ConnectionGene(v.from_node, v.to_node, v.weight, v.innovation)
            c.enabled = v.enabled
            child.connections[k] = c
        return child


def crossover(g1, g2):
    """g1 daha iyi fitness'a sahip olmalı"""
    child = Genome(g1.input_size, g1.output_size, g1.tracker)
    child.nodes       = {}
    child.connections = {}

    all_nodes = set(g1.nodes.keys()) | set(g2.nodes.keys())
    for nid in all_nodes:
        if nid in g1.nodes:
            child.nodes[nid] = NodeGene(nid, g1.nodes[nid].node_type)
        else:
            child.nodes[nid] = NodeGene(nid, g2.nodes[nid].node_type)

    all_innovations = set(g1.connections.keys()) | set(g2.connections.keys())
    for innov in all_innovations:
        in1 = innov in g1.connections
        in2 = innov in g2.connections

        if in1 and in2:
            source = random.choice([g1, g2])
            conn   = source.connections[innov]
        elif in1:
            conn = g1.connections[innov]
        else:
            continue

        new_conn         = ConnectionGene(conn.from_node, conn.to_node, conn.weight, innov)
        new_conn.enabled = conn.enabled

        if in1 and in2:
            if not g1.connections[innov].enabled and not g2.connections[innov].enabled:
                if random.random() < 0.75:
                    new_conn.enabled = False

        child.connections[innov] = new_conn

    return child


def genetic_distance(g1, g2, c1=1.0, c2=0.4):
    all_innovations = set(g1.connections.keys()) | set(g2.connections.keys())
    if not all_innovations:
        return 0.0

    matching        = 0
    weight_diff_sum = 0.0
    disjoint        = 0

    for innov in all_innovations:
        in1 = innov in g1.connections
        in2 = innov in g2.connections
        if in1 and in2:
            matching        += 1
            weight_diff_sum += abs(g1.connections[innov].weight - g2.connections[innov].weight)
        else:
            disjoint += 1

    avg_weight_diff = weight_diff_sum / matching if matching > 0 else 0
    n = max(len(g1.connections), len(g2.connections), 1)

    return (c1 * disjoint / n) + (c2 * avg_weight_diff)


class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members        = []
        self.best_fitness   = 0
        self.stagnation     = 0

    def add(self, genome):
        self.members.append(genome)

    def update_representative(self):
        if self.members:
            self.representative = random.choice(self.members)

    def adjust_fitness(self):
        n = max(len(self.members), 1)
        for genome in self.members:
            genome.fitness = genome.fitness / n


class NEATPopulation:
    def __init__(self, size, input_size, output_size, distance_threshold=3.0):
        self.size               = size
        self.input_size         = input_size
        self.output_size        = output_size
        self.distance_threshold = distance_threshold
        self.tracker            = InnovationTracker()
        self.species            = []
        self.generation         = 0

        self.genomes = [
            Genome(input_size, output_size, self.tracker)
            for _ in range(size)
        ]
        self.speciate()

    def speciate(self):
        for species in self.species:
            species.members = []

        for genome in self.genomes:
            placed = False
            for species in self.species:
                dist = genetic_distance(genome, species.representative)
                if dist < self.distance_threshold:
                    species.add(genome)
                    placed = True
                    break
            if not placed:
                new_species = Species(genome)
                new_species.add(genome)
                self.species.append(new_species)

        self.species = [s for s in self.species if s.members]

        for species in self.species:
            species.update_representative()

    def evolve(self):
        self.generation += 1

        for species in self.species:
            species.adjust_fitness()

        total_fitness = sum(
            sum(g.fitness for g in s.members)
            for s in self.species
        )
        if total_fitness == 0:
            total_fitness = 1

        new_genomes = []

        for species in self.species:
            species_fitness = sum(g.fitness for g in species.members)
            offspring_count = int((species_fitness / total_fitness) * self.size)

            if not offspring_count:
                continue

            best  = max(species.members, key=lambda g: g.fitness)
            elite = best.copy()
            new_genomes.append(elite)

            members_sorted = sorted(species.members, key=lambda g: g.fitness, reverse=True)
            top_members    = members_sorted[:max(1, len(members_sorted)//2)]

            for _ in range(offspring_count - 1):
                if len(top_members) > 1 and random.random() < 0.75:
                    p1, p2 = random.sample(top_members, 2)
                    if p1.fitness < p2.fitness:
                        p1, p2 = p2, p1
                    child = crossover(p1, p2)
                else:
                    child = random.choice(top_members).copy()

                child.mutate()
                new_genomes.append(child)

        while len(new_genomes) < self.size:
            parent = random.choice(self.genomes)
            child  = parent.copy()
            child.mutate()
            new_genomes.append(child)

        self.genomes = new_genomes[:self.size]
        self.speciate()