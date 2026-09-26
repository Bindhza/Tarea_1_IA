import random
import numpy as np

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
            # Verificar si el vecino es transitable (no es un obstáculo) sabiendo que los obstáculos están representados por valores negativos
            if matriz[nuevo_x, nuevo_y] > 0 or matriz[nuevo_x, nuevo_y] == -3:
                costo = 1 if matriz[nuevo_x, nuevo_y] == -3 else matriz[nuevo_x, nuevo_y]
                vecinos.append(((nuevo_x, nuevo_y), costo))  # Costo basado en el valor de la celda
    return vecinos


def propagacion_de_incendio(matriz):
    """
    Propaga el fuego un paso en el tiempo.
    - 0: Muro (no se quema)
    - -1: Fuego
    - > 0: Terreno transitable (susceptible al fuego)
    Retorna una nueva matriz con el fuego propagado.
    """
    # Movimientos posibles para la propagación del fuego (arriba, abajo, izquierda, derecha)
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    nueva_matriz = np.copy(matriz)  # Crear una copia de la matriz original para actualizarla sin afectar la propagación en curso
    
    # Recorrer la matriz para encontrar celdas en llamas
    for i, j in np.ndindex(matriz.shape):
        # celdas en llamas está representada por el valor -1
        if matriz[i, j] == -1:
            
            # lista para almacenar vecinos que pueden ser incendiados
            vecinos_incendiables = []
            
            # añade los vecinos a la lista de vecinos incendiables
            for dx, dy in movimientos:
                nuevo_x, nuevo_y = i + dx, j + dy
                # Verificar si el vecino está dentro de los límites de la matriz y es transitable
                if 0 <= nuevo_x < matriz.shape[0] and 0 <= nuevo_y < matriz.shape[1] and matriz[nuevo_x, nuevo_y] > 0:
                    vecinos_incendiables.append((nuevo_x, nuevo_y)) 
            
            num_a_quemar = min(random.randint(2, 4), len(vecinos_incendiables))  # Determinar cuántos vecinos se quemarán (máximo 4)
            
            # Seleccionar aleatoriamente los vecinos a quemar
            if num_a_quemar > 0:
                elegidos = random.sample(vecinos_incendiables, num_a_quemar)
                for nx, ny in elegidos:
                    nueva_matriz[nx, ny] = -1  # Se queman en la nueva matriz
                    
    return nueva_matriz

# Mapa 1: Alta densidad de obstáculos / Cuello de botella
# Pasillos angostos (ancho 1) que convergen hacia un cuello de botella central que lleva a la salida (-3)
# Propicia alto tráfico y bloqueos locales / callejones sin salida.
mapa1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Cuello de botella angosto (ancho 1)
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0]   # Salida de evacuación
])

# Mapa 2: Densidad media / Laberinto corporativo
# Múltiples salas/oficinas conectadas por intersecciones centrales y pasillos con cruces ciegos y desvíos.
mapa2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0]   # Salida de evacuación
])

# Mapa 3: Baja densidad / Dispersión abierta
# Entorno semiabierto con columnas y tabiques aislados, permitiendo múltiples rutas alternativas hacia la salida.
mapa3 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 0, 0, 0, 0]   # Salida de evacuación
])
