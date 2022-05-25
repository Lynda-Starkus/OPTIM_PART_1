from graph import Graph
import sys
from os import path

# Fichier des couleurs
COLOR_FILE = "colors.txt"
# Fichier d'instance par défaut
GRAPH_FILE_DEFAULT = "datasets/myciel5.col"
# Nombre de noeuds du graphe pour les graphes aléatoires
N = 25
# Générer un graphe aléatoire ou pas
RANDOM_GRAPH = False

# Si un argument est fourni, il est considéré le fichier d'instance du graphe
if len(sys.argv) > 1 and path.exists(sys.argv[1]):
    GRAPH_FILE = sys.argv[1]
else:
    GRAPH_FILE = GRAPH_FILE_DEFAULT


POPLUATION_SIZE = 200
# Pour l'effet de diversification
MUTATION_PROBABILITY = 0.4
# Nombre de générations (itérations) à produire (effectuer) à partir d'une population
GENERATIONS_NUM = 300
# TABU_NUMBER_OF_COLORS = 25
DEBUG = True

# RANDOM_GRAPH = True  -> Générer un graphe aléatoire de taille N
if RANDOM_GRAPH:
    GRAPH = Graph.rand_graph(N)
# RANDOM_GRAPH = False -> Lire le graphe à partir d'un fichier
else:
    GRAPH = Graph.from_file(GRAPH_FILE)
