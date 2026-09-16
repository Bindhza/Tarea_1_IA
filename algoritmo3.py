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
        
        # Generar vecinos (arriba, abajo, izquierda, derecha)
        vecinos = [(celda_actual[0] - 1, celda_actual[1]), (celda_actual[0] + 1, celda_actual[1]),
                    (celda_actual[0], celda_actual[1] - 1), (celda_actual[0], celda_actual[1] + 1)]
        
        for vecino in vecinos:
            # Verificar si el vecino está dentro de los límites de la matriz y es transitable
            if 0 <= vecino[0] < largo and 0 <= vecino[1] < ancho:
                
                # verifica que el vecino no sea un obstáculo y que no haya sido visitado
                # esto teniendo en cuenta que la matriz tiene 0 para celdas transitables y un valor distinto de 0 para obstáculos
                if matriz[vecino[0]][vecino[1]] == 0 and not closed_list[vecino[0]][vecino[1]]:
                    # Calcular los costos g, h y f para el vecino, pero con costo g incrementado en 1, o sea el costo no es variable(a modificar)
                    g_new = cell_details[celda_actual[0]][celda_actual[1]][0] + 1
                    h_new = heuristic(vecino, fin)
                    f_new = g_new + h_new
                    
                    # Si el vecino no ha sido visitado o si el nuevo costo f es menor que el costo f previamente registrado, actualizar la open_list y cell_details
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