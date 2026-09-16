import numpy as np
import heapq

def main():
    

if __name__ == "__main__":
    main()

#algoritmo para recorrer grilla con el algoritmo de Busqueda de Costo Uniforme
def costo_uniforme_busqueda(grilla, inicio, fin):
    matriz = grilla
    
    #inicializar la cola de prioridad con el nodo inicial y costo 0
    cola = [(0, inicio)]
    visitado = {inicio: (0, None)}
    
    while cola:
        #extraer el nodo con el menor costo de la cola
        costo_actual, nodo_actual = heapq.heappop(cola)
        
        #si el nodo actual es el nodo objetivo, reconstruir el camino y retornar
        if nodo_actual == fin:
            costo_uniforme_reconstruccion(visitado, fin)
            return visitado[fin][0], costo_uniforme_reconstruccion(visitado, fin)
        
        for vecino, costo in matriz[nodo_actual]:
            #calcular el costo del vecino
            costo_vecino = costo_actual + costo
            
            #si el vecino no ha sido visitado o se encontró un camino más corto hacia él, no se actualiza el costo
            if vecino not in visitado or costo_vecino < visitado[vecino][0]:
                #actualizar el costo y el nodo padre del vecino en el diccionario de visitados
                visitado[vecino] = (costo_vecino, nodo_actual)
                #agregar el vecino a la cola de prioridad con su costo
                heapq.heappush(cola, (costo_vecino, vecino))
    return None, None  #si no se encuentra un camino hacia el nodo objetivo
        
        
def costo_uniforme_reconstruccion(visitado, fin):
    #reconstruir el camino desde el nodo objetivo hasta el nodo inicial
    reconstruido = []
    nodo_actual = fin
    while nodo_actual is not None:
        reconstruido.append(nodo_actual)
        #encontrar el nodo padre del nodo actual en el diccionario de visitados
        nodo_actual = visitado[nodo_actual][1]
        
    #retornar el camino reconstruido en orden inverso (del nodo inicial al nodo objetivo)
    return reconstruido[::-1]    
