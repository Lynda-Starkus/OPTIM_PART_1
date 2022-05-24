import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

# Classe pour représenter un graphe
class Graph(list):
    SUBPLOT_NUM = 411
    TYPE_COMMENT = "c"
    TYPE_PROBLEM_LINE = "p"
    TYPE_EDGE_DESCRIPTOR = "e"
    EDGE_ON = 1
    EDGE_OFF = 0

    # Retourne le nombre de noeuds du graphe
    def getVerticesNum(self):
        return len(self)

    # Retourne les noeuds voisins du noeud spécifié
    def getNeighbors(self, u):
        return [v for v, j in enumerate(self[u]) if j == Graph.EDGE_ON]

    # Retourne vrai si u et v sont des voisins
    def areNeighbours(self, u, v):
        return u in self.getNeighbors(v)

    def largestDegree(self):
        return max(node.count(Graph.EDGE_ON) for node in self)

    # Dessiner le graphe avec matplotlib
    def draw(
        self,
        title=None,
        draw_func=nx.draw_circular,
        color_key={},
        default_color="lightblue",
        visualize=False,
    ):
        nxgraph = nx.from_numpy_matrix(np.array(self))
        node_color = [color_key.get(v, default_color) for v in nxgraph]
        plt.subplot(Graph.SUBPLOT_NUM, title=title)
        Graph.SUBPLOT_NUM += 1
        draw_func(nxgraph, node_color=node_color, font_weight="bold", with_labels=True)
        if visualize:
            plt.show()

    # Lire et analyser une ligne d'un fichier d'instance de graphe
    @staticmethod
    def parse_line(line):
        if line.startswith(Graph.TYPE_COMMENT):
            return Graph.TYPE_COMMENT, None
        elif line.startswith(Graph.TYPE_PROBLEM_LINE):
            _, _, num_nodes, num_edges = line.split(" ")
            return Graph.TYPE_PROBLEM_LINE, (int(num_nodes), int(num_edges))
        elif line.startswith(Graph.TYPE_EDGE_DESCRIPTOR):
            _, node1, node2 = line.split(" ")
            return Graph.TYPE_EDGE_DESCRIPTOR, (int(node1), int(node2))
        else:
            raise ValueError(f"Unable to parse '{line}'")

    # Générer un graphe à partir d'un fichier d'instance de graphe
    @classmethod
    def from_file(cls, filename):
        matrix = None

        with open(filename) as f:
            problem_set = False
            for line in f.readlines():
                line_type, val = Graph.parse_line(line.strip())
                if line_type == Graph.TYPE_COMMENT:
                    continue
                elif line_type == Graph.TYPE_PROBLEM_LINE and not problem_set:
                    num_nodes, num_edges = val
                    matrix = [
                        [Graph.EDGE_OFF for _ in range(num_nodes)]
                        for _ in range(num_nodes)
                    ]
                    problem_set = True
                elif line_type == Graph.TYPE_EDGE_DESCRIPTOR:
                    if not problem_set:
                        raise RuntimeError("Edge descriptor found before problem line")
                    node1, node2 = val
                    matrix[node1 - 1][node2 - 1] = Graph.EDGE_ON
                    matrix[node2 - 1][node1 - 1] = Graph.EDGE_ON

        return cls(matrix)

    # Générer un graphe aléatoire de taille `n`
    @classmethod
    def rand_graph(cls, n):
        mat = [[Graph.EDGE_OFF for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n, 1):
                mat[i][j] = random.choice([Graph.EDGE_OFF, Graph.EDGE_ON])
                mat[j][i] = mat[i][j]
        return cls(mat)
