import numpy as np
import heapq

def main():
    

if __name__ == "__main__":
    main()

# inicio y fin deben ser tuplas (x, y) representando las coordenadas en la grilla
def greedy_best_first(grilla: np.ndarray, inicio, fin, heuristic):
    matriz = grilla
    # Obtener alto (filas) y ancho (columnas)
    largo, ancho = matriz.shape
    
    #si el inicio es igual al fin, retornar la lista con el inicio
    if inicio == fin:
        return [inicio]
    
    # Donde se almacenan los nodos visitados
    visitados = np.zeros((largo, ancho), dtype=bool)
    
    queue = []
    # Nodo inicial, se agrega a la cola con su valor heurístico
    heapq.heappush(queue, (heuristic(inicio, fin), inicio, []))
    
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

def get_vecinos(matriz, nodo):
    vecinos = []
    x, y = nodo
    # Definir movimientos posibles (arriba, abajo, izquierda, derecha)
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Recorrer los movimientos posibles
    for dx, dy in movimientos:
        
        nuevo_x, nuevo_y = x + dx, y + dy
        
        # Verificar si el vecino está dentro de los límites de la matriz
        if 0 <= nuevo_x < matriz.shape[0] and 0 <= nuevo_y < matriz.shape[1]:
            # Verificar si el vecino es transitable (no es un obstáculo) suponiendo que los obstáculos están representados por valores distintos de 0(cambiar)
            if matriz[nuevo_x, nuevo_y] == 0:
                # Asumiendo costo uniforme de 1(modificar)
                vecinos.append(((nuevo_x, nuevo_y), 1))
    return vecinos