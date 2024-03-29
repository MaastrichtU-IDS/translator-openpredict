from rdf2vec.graph import Vertex
from rdf2vec.walkers import RandomWalker


class WalkletWalker(RandomWalker):
    def __init__(self, depth, walks_per_graph):
        super().__init__(depth, walks_per_graph)

    def extract(self, graph, instances):
        canonical_walks = set()
        for instance in instances:
            walks = self.extract_random_walks(graph, Vertex(str(instance)))
            for walk in walks:
                for n in range(1, len(walk)):
                    canonical_walks.add((walk[0].name, walk[n].name))
        return canonical_walks
