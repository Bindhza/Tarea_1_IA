import numpy as np
from funciones_auxiliar import get_vecinos

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
        return 0, [nodo_actual]
    
    #si se alcanza la profundidad límite, retornar None
    if profundidad_limite <= 0:
        return None, None
    
    #recorre todos los vecinos del nodo actual y después llama recursivamente a la función para cada vecino no visitado
    #terminando la búsqueda si se encuentra el nodo objetivo
    for vecino, peso in get_vecinos(matriz, nodo_actual):
        
        #agregar el vecino a la lista de visitados en caso de que no haya sido visitado previamente
        if vecino not in visitado:
            visitado.append(vecino)
            (costo_hijo, resultado) = dls_recursiva(vecino, fin, profundidad_limite - 1, matriz, visitado)
            #si se encuentra el nodo objetivo, retornar el camino encontrado
            if resultado is not None:
                costo_total = peso + costo_hijo
                camino_total = [nodo_actual] + resultado
                return costo_total, camino_total
                
            visitado.pop()  # Backtracking
            
    return None, None
