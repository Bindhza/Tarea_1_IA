import numpy as np
import heapq

def main():

if __name__ == "__main__":
    main()

def a_star(grilla: np.ndarray, inicio, fin, heuristic):
    matriz = grilla
    # Obtener alto (filas) y ancho (columnas)
    largo, ancho = matriz.shape
    
    if inicio == fin:
        return [inicio]
    
    # Inicializar la lista cerrada con celdas no visitadas
    closed_list = [[False for i in range(ancho)] for j in range(largo)]
    closed_list[inicio[0]][inicio[1]] = True
    
    # Correspondencia de celdas con sus detalles (g, h, f, padre)
    cell_details = [[None for i in range(ancho)] for j in range(largo)]
    
    # Lista con las celdas recorridas y sus costos
    open_list = []
    
    # Inicializar la celda de inicio
    cell_details[inicio[0]][inicio[1]] = (0, heuristic(inicio, fin), heuristic(inicio, fin), None)
    heapq.heappush(open_list, (cell_details[inicio[0]][inicio[1]][2], inicio))
    
    destino_encontrado = False
    
    while len(open_list) > 0:
        # Obtener la celda con el menor costo f
        celda_actual = heapq.heappop(open_list)[1]
        
        # Verificar si se ha llegado al destino
        if celda_actual == fin:
            destino_encontrado = True
            break
            
        # Marcar la celda actual como cerrada al expandirla
        closed_list[celda_actual[0]][celda_actual[1]] = True
        
        # Obtener vecinos válidos y transitables
        for vecino, costo in get_vecinos(matriz, celda_actual):
            # Solo verificar que no haya sido cerrado/visitado
            if not closed_list[vecino[0]][vecino[1]]:
                # g_new ahora usa el costo que entrega get_vecinos
                g_new = cell_details[celda_actual[0]][celda_actual[1]][0] + costo
                h_new = heuristic(vecino, fin)
                f_new = g_new + h_new
                
                if cell_details[vecino[0]][vecino[1]] is None or cell_details[vecino[0]][vecino[1]][2] > f_new:
                    cell_details[vecino[0]][vecino[1]] = (g_new, h_new, f_new, celda_actual)
                    heapq.heappush(open_list, (f_new, vecino))

    # Verificar si se encontró un camino al destino, si no se encontró, retornar None
    if not destino_encontrado:
        return None
    
    # Reconstruir camino desde el fin hacia el inicio
    camino = []
    actual = fin
    
    # Retrocede usando los padres hasta llegar al inicio
    while actual is not None:
        camino.append(actual)
        actual = cell_details[actual[0]][actual[1]][3]  # El índice 3 es el padre
        
    # Invertir para que vaya desde inicio hasta fin
    camino.reverse()

    # almacenar el costo total del camino encontrado
    costo_total = cell_details[fin[0]][fin[1]][0]
    
    return camino

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