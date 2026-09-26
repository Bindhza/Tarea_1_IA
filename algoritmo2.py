import numpy as np
from funciones_auxiliar import get_vecinos

def busqueda_profundidad_limitada(grilla, inicio, fin, limite):
    
    # matriz inicializada con la grilla de entrada
    matriz = grilla
    
    # nodo inicial y conjunto de nodos visitados en la rama actual
    visitado = set([inicio])
    # diccionario para podar ramas redundantes que ya fueron exploradas con profundidad igual o mayor
    mejor_limite = {}
    
    # llamar a la función recursiva de búsqueda en profundidad limitada
    return dls_recursiva(inicio, fin, limite, matriz, visitado, mejor_limite)

# algoritmo de función recursiva para la búsqueda en profundidad limitada con poda
def dls_recursiva(nodo_actual, fin, profundidad_limite, matriz, visitado, mejor_limite):
    
    # si el nodo actual es el nodo objetivo, retornar el camino encontrado
    if nodo_actual == fin:
        return 0, [nodo_actual]
    
    # si se alcanza la profundidad límite, retornar None
    if profundidad_limite <= 0:
        return None, None
    
    # Poda: si este nodo ya fue explorado con un límite restante mayor o igual, no repetir la exploración
    if mejor_limite.get(nodo_actual, -1) >= profundidad_limite:
        return None, None
    mejor_limite[nodo_actual] = profundidad_limite
    
    # recorre todos los vecinos del nodo actual y después llama recursivamente a la función para cada vecino no visitado
    # terminando la búsqueda si se encuentra el nodo objetivo
    for vecino, peso in get_vecinos(matriz, nodo_actual):
        
        # agregar el vecino al camino actual si no está en él
        if vecino not in visitado:
            visitado.add(vecino)
            costo_hijo, resultado = dls_recursiva(vecino, fin, profundidad_limite - 1, matriz, visitado, mejor_limite)
            # si se encuentra el nodo objetivo, retornar el camino encontrado
            if resultado is not None:
                costo_total = peso + costo_hijo
                camino_total = [nodo_actual] + resultado
                return costo_total, camino_total
                
            visitado.remove(vecino)  # Backtracking
            
    return None, None
