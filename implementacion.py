from algoritmo1 import costo_uniforme_busqueda
from algoritmo2 import busqueda_profundidad_limitada
from algoritmo3 import a_star
from algoritmo4 import greedy_best_first
from algoritmo5 import algoritmo_genetico
from funciones_auxiliar import propagacion_de_incendio, mapa1, mapa2, mapa3
import numpy as np
import random

def iniciar_fuego(matriz, posiciones_excluidas=None):
    """
    Inicia el fuego en la matriz en una posición aleatoria que sea transitable.
    - 0: Muro (no se quema)
    - -1: Fuego
    - > 0: Terreno transitable (susceptible al fuego)
    Retorna una nueva matriz con el fuego iniciado.
    """
    nueva_matriz = np.copy(matriz)  # Crear una copia de la matriz original
    
    # Obtener las posiciones transitables
    posiciones_transitables = np.argwhere(nueva_matriz > 0)
    
    if posiciones_excluidas is not None:
        # Excluir las posiciones de inicio de los agentes para no quemarlos al segundo 0
        posiciones_validas = [
            tuple(posicion) for posicion in posiciones_transitables 
            if tuple(posicion) not in posiciones_excluidas
        ]
    else:
        posiciones_validas = [tuple(posicion) for posicion in posiciones_transitables]
    
    if len(posiciones_validas) == 0:
        raise ValueError("No hay posiciones transitables para iniciar el fuego.")
    
    # Seleccionar una posición aleatoria para iniciar el fuego
    posicion_fuego = random.choice(posiciones_validas)
    
    # Iniciar el fuego en la posición seleccionada
    nueva_matriz[posicion_fuego[0], posicion_fuego[1]] = -1
    
    return nueva_matriz

def simular_profundidad_limitada(mapa, posiciones_iniciales, meta=None, limite_profundidad=100):
    """
    Ejecuta la simulación de evacuación con Búsqueda en Profundidad Limitada.
    
    Parámetros convenientes:
    - mapa: Matriz del entorno (mapa1, mapa2, mapa3, etc.)
    - posiciones_iniciales: Lista de coordenadas iniciales de los agentes [(x, y), ...]
    - meta: Coordenada de la salida (si es None, busca automáticamente el -3 en el mapa)
    - limite_profundidad: Cota máxima de profundidad para el algoritmo DLS (por defecto 100)
    """
    print("Ejecutando Búsqueda en Profundidad Limitada...")
    
    # Auto-detectar la salida (-3) en el mapa si no se especifica
    if meta is None:
        salida_coords = np.argwhere(mapa == -3)
        if len(salida_coords) > 0:
            meta = (int(salida_coords[0][0]), int(salida_coords[0][1]))
        else:
            meta = (14, 7)
    
    # Copia sobre la que se va a trabajar para no modificar el mapa original
    mapa_simulacion = mapa.copy()
    mapa_simulacion = iniciar_fuego(mapa_simulacion, posiciones_iniciales)  # Inicia el fuego en una posición aleatoria
    mapa_simulacion_agentes = None  # None para que calcule el camino en la primera iteración
    
    # Inicializar dinámicamente los agentes según las posiciones recibidas: (Vivo, x, y, Termino, Camino, Costo)
    agenetes_profundidad_limitada = [(True, pos[0], pos[1], False, [], 0) for pos in posiciones_iniciales]
    caminos_recorridos = [[(pos[0], pos[1])] for pos in posiciones_iniciales]
    
    bucle_profundidad_limitada = True
    contador_propagacion = 0  # Contador para controlar la propagación del fuego
    
    # Bucle principal de simulación
    while bucle_profundidad_limitada:
        
        # Lógica de propagación de fuego por el mapa
        if contador_propagacion >= random.randint(1, 3):  # Propaga el fuego cada 1 o 3 iteraciones
            contador_propagacion = 0  # Reiniciar el contador
            mapa_simulacion = propagacion_de_incendio(mapa_simulacion)
            
            # Verificar si el fuego recién propagado alcanzó a algún agente en su posición actual
            for i, agente in enumerate(agenetes_profundidad_limitada):
                if agente[0] and not agente[3]:
                    agente_pos_x, agente_pos_y = agente[1], agente[2]
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        # El fuego alcanzó al agente en su casilla -> muere y termina
                        agenetes_profundidad_limitada[i] = (False, agente_pos_x, agente_pos_y, True, [], agente[5])
                        print(f"Agente {i} fue alcanzado por el fuego en su posición actual ({agente_pos_x}, {agente_pos_y}) y murió.")
        else:
            contador_propagacion += 1
        
        # Mide si la matriz cambió para que los agentes puedan buscar nuevos caminos
        cambio_matriz = mapa_simulacion_agentes is None or not np.array_equal(mapa_simulacion, mapa_simulacion_agentes)
        if cambio_matriz:
            # Actualizar la copia del mapa para reflejar los cambios
            mapa_simulacion_agentes = mapa_simulacion.copy()
            for i, agente in enumerate(agenetes_profundidad_limitada):
                if agente[0] and not agente[3]:  # Si el agente está vivo y no ha terminado
                    # Buscar un nuevo camino hacia la meta con el límite de profundidad indicado
                    costo_algoritmo, nuevo_camino = busqueda_profundidad_limitada(mapa_simulacion_agentes, (agente[1], agente[2]), meta, limite_profundidad)
                    # Actualizar el camino restante del agente
                    if nuevo_camino is not None:
                        camino_pasos = nuevo_camino[1:]
                    else:
                        camino_pasos = []  # No hay camino disponible (bloqueado por fuego u obstáculos)
                    
                    # Mantenemos su costo acumulado real agente[5]
                    agenetes_profundidad_limitada[i] = (agente[0], agente[1], agente[2], agente[3], camino_pasos, agente[5])
        
        # Sigue los movimientos del camino proporcionado por el algoritmo
        # y verifica si el agente ha llegado a su destino o si ha muerto por el fuego
        for i, agente in enumerate(agenetes_profundidad_limitada):
            vivo, agente_pos_x, agente_pos_y, termino, camino, costo_acumulado = agente
            if vivo and not termino:
                if camino:  # Si hay un camino disponible
                    siguiente_posicion = camino.pop(0) 
                    
                    # Verificar si la siguiente posición es válida
                    if siguiente_posicion == (None, None):
                        continue
                    
                    siguiente_pos_x, siguiente_pos_y = siguiente_posicion
                    caminos_recorridos[i].append(siguiente_posicion)
                    
                    # Sumar el costo real de entrar a esta nueva casilla
                    costo_paso = 1 if mapa[siguiente_pos_x, siguiente_pos_y] == -3 else int(mapa[siguiente_pos_x, siguiente_pos_y])
                    nuevo_costo = costo_acumulado + costo_paso
                    
                    # Verificar si la siguiente posición tiene fuego
                    if mapa_simulacion[siguiente_pos_x, siguiente_pos_y] == -1:
                        agenetes_profundidad_limitada[i] = (False, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)  # Agente muere
                        print(f"Agente {i} avanzó a ({siguiente_pos_x}, {siguiente_pos_y}), entró al fuego y murió.")
                    # Verificar si la siguiente posición es la meta
                    elif siguiente_posicion == meta:
                        agenetes_profundidad_limitada[i] = (True, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)  # Agente llega a la meta
                        print(f"¡Agente {i} escapó exitosamente por la salida en {meta}!")
                    else:
                        agenetes_profundidad_limitada[i] = (agente[0], siguiente_pos_x, siguiente_pos_y, False, camino, nuevo_costo)  # Continúa avanzando
                else:
                    # Si no hay camino disponible, el agente NO termina: espera en su posición a que el fuego lo alcance
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        agenetes_profundidad_limitada[i] = (False, agente_pos_x, agente_pos_y, True, [], costo_acumulado)
                        print(f"Agente {i} atrapado en ({agente_pos_x}, {agente_pos_y}) murió por el fuego.")
        
        # Condición para terminar el bucle: se detiene solo cuando todos hayan escapado o hayan muerto
        bucle_profundidad_limitada = any(agente[0] and not agente[3] for agente in agenetes_profundidad_limitada)
    
    print("Búsqueda en Profundidad Limitada finalizada.")
    print("#"*50)
    # Muestra el estado de cada agente
    for i, agente in enumerate(agenetes_profundidad_limitada):
        if agente[0] and (agente[1], agente[2]) == meta:
            estado_texto = "Vivo (Escapó)"
        else:
            estado_texto = "Muerto por el fuego"
        print(f"Agente {i}: Posición final=({agente[1]}, {agente[2]}), Estado={estado_texto}, Vivo={agente[0]}, Terminó={agente[3]}")
        # Camino recorrido por el agente
        print(f"Agente {i}: Camino recorrido={caminos_recorridos[i]}, Costo del camino={agente[5]}")
        print("-"*50)
        
    # Imprime el estado final del mapa
    print("Estado final del mapa:")
    print(mapa_simulacion)
    print("="*50)
    
    return agenetes_profundidad_limitada, caminos_recorridos, mapa_simulacion

def posiciones_iniciales_aleatorias(mapa, num_agentes):
    """
    Genera posiciones iniciales aleatorias para los agentes en el mapa.
    - mapa: Matriz del entorno (mapa1, mapa2, mapa3, etc.)
    - num_agentes: Número de agentes a colocar
    Retorna una lista de coordenadas iniciales [(x, y), ...]
    """
    # Obtener todas las posiciones transitables en el mapa
    posiciones_transitables = np.argwhere(mapa > 0)
    
    # Verificar que haya suficientes posiciones transitables para colocar a todos los agentes
    if len(posiciones_transitables) < num_agentes:
        raise ValueError("No hay suficientes posiciones transitables para colocar a todos los agentes.")
    
    posiciones_aleatorias = random.sample(list(map(tuple, posiciones_transitables)), num_agentes)
    
    return posiciones_aleatorias

def main():
    # Ejecutar la simulación pasándole el mapa y las posiciones de los agentes
    posiciones = posiciones_iniciales_aleatorias(mapa1, 3)
    # posiciones = [(1, 1), (8, 1), (11, 1)]  # Posiciones iniciales fijas para pruebas
    
    
    simular_profundidad_limitada(mapa1, posiciones)

if __name__ == '__main__':
    main()
