import numpy as np
import heapq
from funciones_auxiliar import get_vecinos

# inicio y fin deben ser tuplas (x, y) representando las coordenadas en la grilla
def greedy_best_first(grilla: np.ndarray, inicio, fin, heuristic):
    matriz = grilla
    # Obtener alto (filas) y ancho (columnas)
    largo, ancho = matriz.shape
    
    #si el inicio es igual al fin, retornar la lista con el inicio
    if inicio == fin:
        return 0, [inicio]
    
    # Donde se almacenan los nodos visitados
    visitados = np.zeros((largo, ancho), dtype=bool)
    
    queue = []
    # Nodo inicial, se agrega a la cola con su valor heurístico
    heapq.heappush(queue, (heuristic(inicio, fin), inicio, 0, []))
    
    # Se ejecuta mientras haya nodos en la cola
    while queue:
        h, actual, recorridos = heapq.heappop(queue)
        
        if actual == fin:
            return recorridos + [actual]
        
        # Agregar el nodo actual a la lista de visitados
        if not visitados[actual]:
            visitados[actual] = True
            
            # Obtener vecinos del nodo actual
            for vecino, costo in get_vecinos(matriz, actual):
                if not visitados[vecino]:
                    heapq.heappush(queue, (heuristic(vecino, fin), vecino, recorridos + [actual]))
    return None
