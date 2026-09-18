import numpy as np
def main():
    

if __name__ == "__main__":
    main()


def busqueda_profundidad_limitada(grilla, inicio, fin, limite):
    
    #matriz inicializada con la grilla de entrada
    matriz = grilla
    
    #nodo inicial y lista de nodos visitados
    visitado = [inicio]
    
    #llamar a la función recursiva de búsqueda en profundidad limitada
    return dls_recursiva(inicio, fin, limite, matriz, visitado)

#algoritmo de función recursiva para la búsqueda en profundidad limitada
def dls_recursiva(nodo_actual, fin, profundidad_limite, matriz, visitado):
    
    #si el nodo actual es el nodo objetivo, retornar el camino encontrado
    if nodo_actual == fin:
        return [nodo_actual]
    
    #si se alcanza la profundidad límite, retornar None
    if profundidad_limite <= 0:
        return None
    
    #recorre todos los vecinos del nodo actual y después llama recursivamente a la función para cada vecino no visitado
    #terminando la búsqueda si se encuentra el nodo objetivo
    for vecino, peso in get_vecinos(matriz, nodo_actual):
        
        #agregar el vecino a la lista de visitados en caso de que no haya sido visitado previamente
        if vecino not in visitado:
            visitado.append(vecino)
            resultado = dls_recursiva(vecino, fin, profundidad_limite - 1, matriz, visitado)
            #si se encuentra el nodo objetivo, retornar el camino encontrado
            if resultado is not None:
                return [nodo_actual] + resultado
            visitado.pop()
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