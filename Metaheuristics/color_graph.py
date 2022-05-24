from config import (
    GRAPH,
    COLOR_FILE,
    DEBUG,
)

from graph import Graph
import matplotlib.pyplot as plt
import time
import random
from collections import deque
from sys import stderr


def debugPrint(*args, debug=DEBUG):
    if debug:
        print("[DEBUG] ", end="", flush=True)
        print(*args, file=stderr)


# Retourne le nombre de couleurs utilisé pour la coloration du graphe
def nbColorsUsed(coloration):
    return len(set(coloration))


# Algorithme de coloration de graphe en utilisant la métaheuristique RT
def greedyColoring(graph):
    numVertices = graph.getVerticesNum()
    # Coloration du graphe (résultat)
    coloring = [-1] * numVertices

    # Assigner la première couleur au premier nœud
    coloring[0] = 0
    available = [False] * numVertices

    # Assigner des couleurs aux nœuds restants
    for u in range(numVertices):
        # Traiter tous les nœuds adjacents et marquer leurs couleurs comme non disponibles
        neighbors = graph.getNeighbors(u)
        for i in neighbors:
            if coloring[i] != -1:
                available[coloring[i]] = True

        # Trouver la première couleur disponible
        cr = 0
        while cr < numVertices:
            if available[cr] == False:
                break
            cr += 1

        # Assigner la couleur trouvée
        coloring[u] = cr
        # Remettre les valeurs à faux pour la prochaine itération
        for i in neighbors:
            if coloring[i] != -1:
                available[coloring[i]] = False

    return coloring


class gcpGA:
    def __init__(self, graph: Graph, pop_size, mut_proba, num_gen):
        # Pour les paramètres GA
        self.population_size = pop_size
        self.mutation_probability = mut_proba
        self.number_generations = num_gen

        assert num_gen > 0
        # Pour le graphe
        self.graph = graph
        self.num_vertices = self.graph.getVerticesNum()
        self.num_colors = self._upper_bound()

    """Retourne le minimum entre un nombre de couleurs générées par greedyColoring et le plus grand degré + 1"""

    def _upper_bound(self):
        return min(
            nbColorsUsed(greedyColoring(self.graph)), self.graph.largestDegree() + 1
        )

    """Create individual"""
    """Retourne aléatoirement un individu (une solution pas forcément valide)"""

    def _create_individual(self):
        return [random.randint(1, self.num_colors) for _ in range(self.num_vertices)]

    """Fitness"""
    """Retourne le nombre de conflits générés par une solution (individu)"""

    def _fitness(self, individual):
        return sum(
            self._in_conflict(i, j, individual)
            for i in range(self.num_vertices)
            for j in range(i, self.num_vertices)
        )

    """Retourne vrai si deux nœuds voisins sont colorés avec la même couleur"""

    def _in_conflict(self, u, v, individual):
        return individual[u] == individual[v] and self.graph.areNeighbours(u, v)

    """Mutation"""

    def _mutation(self, individual):
        check = random.uniform(0, 1)
        if check <= self.mutation_probability:
            pos = random.randint(0, len(individual) - 1)
            individual[pos] = random.randint(1, self.num_colors)
        return individual

    """Crossover"""

    def _crossover(self, parent1, parent2, start=3):
        pos = random.randint(start, len(parent1) - 1)
        child1 = parent1[:pos] + parent2[pos:]
        child2 = parent2[:pos] + parent1[pos:]
        return child1, child2

    """Generate population"""
    """Retourne un ensemble d'individus aléatoires"""

    def _gen_population(self):
        return [self._create_individual() for _ in range(self.population_size)]

    """Generate a new population from an old one"""

    def _gen_new_pop_from_old(self, old_population):
        new_population = []
        random.shuffle(old_population)
        for i in range(0, self.population_size - 1, 2):
            child1, child2 = self._crossover(old_population[i], old_population[i + 1])
            new_population.extend((child1, child2))
        return new_population

    """Tournament Selection"""  # Sélection par tournoi

    def _tournament_selection(self, population):
        new_population = []
        for j in range(2):
            random.shuffle(population)
            for i in range(0, self.population_size - 1, 2):
                if self._fitness(population[i]) < self._fitness(population[i + 1]):
                    new_population.append(population[i])
                else:
                    new_population.append(population[i + 1])
        return new_population

    """GA"""
    """Retourne la meilleure coloration trouvée"""

    def exec(self):
        stop = False
        coloring = []
        while not stop and self.num_colors > 0:
            # Essayer de trouver une solution valide avec k = self.num_colors couleurs
            # fitness, best_indiv, gen = self.ga(stop_criteria = 0)
            population = self._gen_population()

            # Initialiser best_fitness et fittest_individual
            best_fitness = self._fitness(population[0])
            fittest_individual = population[0]
            generation = 0
            while best_fitness != 0 and generation != self.number_generations:
                generation += 1

                # Sélection
                population = self._tournament_selection(population)

                # Crossover
                new_population = self._gen_new_pop_from_old(population)

                # Mutation
                # Mise à jour de la population comme étant les nouveaux indidivus générés par le croisement puis modifier certains avec la mutation pour plus de diversification
                population = [
                    self._mutation(individual) for individual in new_population
                ]

                # Mettre à jour best_fitness et fittest_individual
                best_fitness, idx = min(
                    (a, b) for b, a in enumerate(map(self._fitness, population))
                )
                fittest_individual = population[idx]

                if generation % 100 == 0:
                    debugPrint(
                        f"Generation: {generation}, Best_Fitness: {best_fitness}, Individual: {fittest_individual}"
                    )

            if best_fitness != 0:
                # Si la boucle retourne fitness != 0 c'est qu'on a atteint le nombre max d'itération sans trouver une solution valide,
                # l'exécution s'arrête et la solution est le nombre de couleurs trouvées dans l'itération précédente
                stop = True
                self.num_colors += 1
            else:
                # Si fitness == 0 alors la coloration à k couleurs est possible
                debugPrint(
                    f"Le graphe est {self.num_colors}-colorable après {generation} générations"
                )
                coloring = fittest_individual
                # Essayer de trouver une solution avec k - 1 couleurs pour la prochaine itération
                self.num_colors -= 1

        # On sort de l'algorithme une fois qu'on trouve une coloration possible à k couleurs et celle à k-1 est impossible => nb chromatique = k
        return coloring


# Fonction wrapper sur la fonction exec de la classes gcpGA
def gcpGAWrapper(graph):
    return gcpGA(graph, POPLUATION_SIZE, MUTATION_PROBABILITY, GENERATIONS_NUM).exec()


# Retourne une liste de couleurs à partir d'un fichier texte de couleurs
def parse_colors(color_file, randomize=False):
    with open(color_file) as f:
        colors = f.read().split("\n")
    randomize and random.shuffle(colors)
    return colors


# Afficher les résultats d'un algorithme
def printAlgResults(algname, nbColors, coloration, timeTaken):
    print(f"* Solution obtenue avec l'algorithme '{algname}' :")
    print(f"\t- Nombre de couleurs utilisées : {nbColors}")
    print(f"\t- Coloration : {coloration}")

    print(f"\t- Temps pris par l'algorithme : {timeTaken} seconde(s)\n")


# Générer les couleurs pour l'affichage à partir d'une coloration
def generateColorKey(coloration, colors=None, colorfile=COLOR_FILE):
    if colors is None:
        colors = parse_colors(colorfile)
    return {i: colors[color_idx] for i, color_idx in enumerate(coloration)}


# Exécuter un algorithme de coloration sur un graphe
def execColoringAlgorithm(graph, algname, algfunc, *args, **kwargs):
    print(f"* Démarrage de l'algorithme '{algname}'")
    # Démarrer le minuteur
    t = time.time()
    # Exécuter l'algorithme de coloration
    coloration = algfunc(graph, *args, **kwargs)
    # Mesure du temps pris par l'algorithme
    dt = time.time() - t
    # Nombre de couleurs utilisé par la coloration
    nbColors = nbColorsUsed(coloration)
    # Afficher les résultats de l'algorithme
    printAlgResults(algname, nbColors, coloration, dt)
    # Génération des couleurs
    color_key = generateColorKey(coloration)
    # Afficher le résultat graphiquement
    title = f"Graphe coloré avec l'algorithme '{algname}'"
    GRAPH.draw(color_key=color_key, title=title)


if __name__ == "__main__":
    GRAPH.draw(title="Graphe en entrée")

    POPLUATION_SIZE = 200
    GENERATIONS_NUM = 300
    MUTATION_PROBABILITY = 0.4 
  

    # Exécuter l'algorithme GA
    execColoringAlgorithm(GRAPH, "GA", gcpGAWrapper)
    