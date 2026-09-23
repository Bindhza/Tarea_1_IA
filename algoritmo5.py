import numpy as np
import random
from funciones_auxiliar import get_vecinos

# algoritmo para recorrer grilla con Algoritmo Genético
def algoritmo_genetico(grilla, inicio, fin, poblacion_inicial, num_generaciones, tasa_mutacion):
    # Implementación del algoritmo genético para encontrar un camino en la grilla
    matriz = grilla
    # Obtener alto (filas) y ancho (columnas)
    largo, ancho = matriz.shape        
    

    if inicio == fin:
        return 0, [inicio]
    
    # Longitud máxima de movimientos por cada individuo
    longitud_camino = largo * ancho * 2
    
    # Si poblacion_inicial no es una lista, se genera la población inicial aleatoria
    if isinstance(poblacion_inicial, int):
        # Crear población inicial aleatoria
        poblacion = create_poblacion_inicial(poblacion_inicial, longitud_camino)
    else:
        # Si se proporciona una población inicial, se utiliza directamente
        poblacion = list(poblacion_inicial)
        
    
    for generacion in range(num_generaciones):
        # Evaluar el fitness de cada individuo en la grilla
        fitnesses = [calcular_fitness(ind, matriz, inicio, fin) for ind in poblacion]
        
        # Encontrar el mejor individuo de la generación actual
        indice_mejor = max(range(len(poblacion)), key=lambda i: fitnesses[i])
        mejor_individuo = poblacion[indice_mejor]
        camino_mejor, llego = simular_camino(mejor_individuo, matriz, inicio, fin)
        
        # Si el mejor individuo llegó al objetivo, retornar el costo y el camino
        if llego:
            costo_total = len(camino_mejor) - 1
            return costo_total, camino_mejor
            
        # Selección por torneo para elegir padres para la siguiente generación
        padres = seleccion(poblacion, fitnesses)
        
        # conservar el mejor individuo directamente en la nueva generación
        nueva_poblacion = [mejor_individuo]
        
        # Generar descendencia mediante cruce y mutación
        while len(nueva_poblacion) < len(poblacion):
            padre1 = random.choice(padres)
            padre2 = random.choice(padres)
            
            hijo1, hijo2 = crossover(padre1, padre2)
            
            nueva_poblacion.append(mutacion(hijo1, tasa_mutacion))
            if len(nueva_poblacion) < len(poblacion):
                nueva_poblacion.append(mutacion(hijo2, tasa_mutacion))
                
        poblacion = nueva_poblacion
        
    # Si no se encontró un camino tras todas las generaciones
    return None, None

# Posibles movimientos en la grilla (arriba, abajo, izquierda, derecha)
movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# funcion para simular el recorrido de un individuo en la grilla
def simular_camino(individuo, matriz, inicio, fin):
    camino = [inicio]
    actual = inicio
    
    for dx, dy in individuo:
        nuevo_x, nuevo_y = actual[0] + dx, actual[1] + dy
        
        # Verificar límites de la grilla
        if 0 <= nuevo_x < matriz.shape[0] and 0 <= nuevo_y < matriz.shape[1]:
            # Verificar si la celda es transitable  si el valor es mayor o igual a 0, se considera transitable
            if matriz[nuevo_x, nuevo_y] >= 0:
                actual = (nuevo_x, nuevo_y)
                camino.append(actual)
                if actual == fin:
                    return camino, True
                    
    return camino, False

# funcion para calcular el fitness del individuo basado en su camino simulado, usando manhattan distance
def calcular_fitness(individuo, matriz, inicio, fin):
    camino, llego = simular_camino(individuo, matriz, inicio, fin)
    pos_final = camino[-1]
    
    # Distancia Manhattan hacia el objetivo
    distancia = abs(pos_final[0] - fin[0]) + abs(pos_final[1] - fin[1])
    
    if llego:
        # Recompensa alta por llegar a la meta y favorecer caminos más cortos
        return 1000.0 + (1000.0 / len(camino))
    else:
        # Mayor fitness mientras más cerca termine del objetivo
        return 1.0 / (distancia + 1.0)

# funcion para crear la población inicial
def create_poblacion_inicial(tamano_poblacion, longitud_camino):
    poblacion = []
    for _ in range(tamano_poblacion):
        # Cada individuo es una secuencia aleatoria de pasos/direcciones
        individuo = [random.choice(movimientos) for i in range(longitud_camino)]
        poblacion.append(individuo)
    return poblacion

# funcion de selección por torneo
def seleccion(poblacion, fitnesses, tamano_torneo=3):
    selected = []
    tamano_torneo = min(tamano_torneo, len(poblacion))
    poblacion_fitness = list(zip(poblacion, fitnesses))
    for _ in range(len(poblacion)):
        torneo = random.sample(poblacion_fitness, tamano_torneo)
        winner = max(torneo, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected

# funcion de crossover (cruce de un punto) para combinar dos caminos
def crossover(padre1, padre2):
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2

# funcion de mutación para variar aleatoriamente direcciones en el camino
def mutacion(individuo, tasa_mutacion):
    individuo_mutado = list(individuo)
    for i in range(len(individuo_mutado)):
        if random.random() < tasa_mutacion:
            individuo_mutado[i] = random.choice(movimientos)
    return individuo_mutado