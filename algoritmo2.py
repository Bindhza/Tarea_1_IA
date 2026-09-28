import numpy as np
import random
import time
from collections import deque
from funciones_auxiliar import (
    get_vecinos, 
    propagacion_de_incendio, 
    iniciar_fuego, 
    mapa1, 
    mapa2, 
    mapa3,
    posiciones_iniciales_aleatorias,
    calcular_costo_penalizado,
    generar_mapa_con_congestion,
    POSICIONES_AGENTES_ESTANDAR,
    POSICION_FUEGO_ESTANDAR,
    CAPACIDAD_MAXIMA_ESTANDAR,
    obtener_posiciones_estandar,
    calcular_metricas_simulacion,
    calcular_e_imprimir_metricas
)

# Algoritmo de Búsqueda en Amplitud (BFS) - Búsqueda No Informada
def busqueda_amplitud(grilla, inicio, fin):
    """
    Búsqueda en Amplitud (BFS) para encontrar el camino más corto en cantidad de pasos.
    Garantiza encontrar la ruta con menor número de casillas transitadas.
    Retorna: (costo_acumulado, camino) o (None, None) si no hay camino.
    """
    matriz = grilla
    if inicio == fin:
        return 0, [inicio]
        
    cola = deque([inicio])
    # padres: {nodo: (nodo_padre, costo_arista)}
    padres = {inicio: (None, 0)}
    destino_encontrado = False
    
    while cola: 
        nodo_actual = cola.popleft()
        
        if nodo_actual == fin:
            destino_encontrado = True
            break
            
        for vecino, costo in get_vecinos(matriz, nodo_actual):
            if vecino not in padres:
                padres[vecino] = (nodo_actual, costo)
                cola.append(vecino)
                if vecino == fin:
                    destino_encontrado = True
                    break
        if destino_encontrado:
            break
            
    if not destino_encontrado and fin not in padres:
        return None, None
        
    # Reconstrucción del camino desde fin hasta inicio
    camino = []
    actual = fin
    costo_total = 0
    while actual is not None:
        camino.append(actual)
        padre, costo = padres[actual]
        costo_total += costo
        actual = padre
        
    camino.reverse()
    return round(float(costo_total), 2), camino


# Alias de compatibilidad
busqueda_profundidad_limitada = busqueda_amplitud


def simular_amplitud(mapa, cantidad_agentes=80, meta=None, capacidad_maxima=4, posicion_fuego=None, **kwargs):
    """
    Ejecuta la simulación de evacuación con Búsqueda en Amplitud (BFS).
    - mapa: Matriz del entorno (mapa1, mapa2, mapa3, etc.)
    - cantidad_agentes: Puede ser un entero o una lista de coordenadas iniciales
    - meta: Coordenada de la salida (si es None, busca automáticamente el -3 en el mapa)
    - capacidad_maxima: Cantidad máxima de agentes permitidos por casilla (por defecto 4)
    - posicion_fuego: Coordenada inicial fija del fuego (si es None, usa POSICION_FUEGO_ESTANDAR)
    Retorna un diccionario con las métricas de la simulación.
    """
    if meta is None:
        salida_coords = np.argwhere(mapa == -3)
        if len(salida_coords) > 0:
            meta = (int(salida_coords[0][0]), int(salida_coords[0][1]))
        else:
            meta = (mapa.shape[0] - 1, mapa.shape[1] // 2)
            
    if isinstance(cantidad_agentes, int):
        posiciones_iniciales = obtener_posiciones_estandar(mapa, cantidad_agentes)
    elif cantidad_agentes is None:
        posiciones_iniciales = list(POSICIONES_AGENTES_ESTANDAR)
    else:
        posiciones_iniciales = list(cantidad_agentes)
    
    mapa_simulacion = mapa.copy()
    mapa_simulacion = iniciar_fuego(mapa_simulacion, posiciones_iniciales, posicion_fuego=posicion_fuego)
    mapa_simulacion_agentes = None
    
    # Inicializar agentes: (Vivo, x, y, Termino, Camino, Costo)
    agentes_bfs = [(True, pos[0], pos[1], False, [], 0) for pos in posiciones_iniciales]
    caminos_recorridos = [[(pos[0], pos[1])] for pos in posiciones_iniciales]
    turnos_escape = [None] * len(posiciones_iniciales)
    
    # Registro de ocupación de las casillas por agentes activos
    ocupacion = {}
    for pos in posiciones_iniciales:
        ocupacion[pos] = ocupacion.get(pos, 0) + 1
    
    bucle_bfs = True
    contador_propagacion = 0
    iteracion = 0
    max_iteraciones = 250
    
    # Bucle principal de simulación
    while bucle_bfs:
        iteracion += 1
        if iteracion >= max_iteraciones:
            for i, agente in enumerate(agentes_bfs):
                if agente[0] and not agente[3]:
                    agentes_bfs[i] = (False, agente[1], agente[2], True, [], agente[5])
            break
        
        # Propagación de fuego por el mapa cada k turnos discretos
        fuego_avanzo = False
        if contador_propagacion >= random.randint(2, 3):
            contador_propagacion = 0
            mapa_con_fuego_nuevo = propagacion_de_incendio(mapa_simulacion)
            fuego_avanzo = not np.array_equal(mapa_con_fuego_nuevo, mapa_simulacion)
            mapa_simulacion = mapa_con_fuego_nuevo
            
            # Verificar si el fuego alcanzó a algún agente en su posición actual
            for i, agente in enumerate(agentes_bfs):
                if agente[0] and not agente[3]:
                    agente_pos_x, agente_pos_y = agente[1], agente[2]
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_bfs[i] = (False, agente_pos_x, agente_pos_y, True, [], agente[5])
        else:
            contador_propagacion += 1
        
        # Recalcular caminos si el mapa cambió por avance del fuego
        cambio_matriz = mapa_simulacion_agentes is None or not np.array_equal(mapa_simulacion, mapa_simulacion_agentes)
        if cambio_matriz:
            mapa_simulacion_agentes = mapa_simulacion.copy()
            mapa_con_congestion = generar_mapa_con_congestion(mapa_simulacion_agentes, ocupacion, meta, capacidad_maxima=capacidad_maxima)
            cache_caminos = {}
            for i, agente in enumerate(agentes_bfs):
                if agente[0] and not agente[3]:
                    pos_ag = (agente[1], agente[2])
                    if pos_ag not in cache_caminos:
                        costo_algoritmo, nuevo_camino = busqueda_amplitud(mapa_con_congestion, pos_ag, meta)
                        cache_caminos[pos_ag] = nuevo_camino
                    else:
                        nuevo_camino = cache_caminos[pos_ag]
                        
                    if nuevo_camino is not None:
                        camino_pasos = list(nuevo_camino[1:])
                    else:
                        camino_pasos = []
                    
                    agentes_bfs[i] = (agente[0], agente[1], agente[2], agente[3], camino_pasos, agente[5])
        
        # Movimiento discreto y acción de esperar
        for i, agente in enumerate(agentes_bfs):
            vivo, agente_pos_x, agente_pos_y, termino, camino, costo_acumulado = agente
            if vivo and not termino:
                if camino:
                    siguiente_posicion = camino.pop(0)
                    if siguiente_posicion == (None, None):
                        continue
                    
                    siguiente_pos_x, siguiente_pos_y = siguiente_posicion
                    
                    # Verificar si la casilla destino alcanzó la capacidad física máxima
                    densidad_destino = ocupacion.get(siguiente_posicion, 0)
                    if siguiente_posicion != meta and densidad_destino >= capacidad_maxima:
                        # Casilla no transitable por capacidad máxima: el agente espera en su posición actual
                        camino.insert(0, siguiente_posicion)
                        continue
                    
                    # Liberar casilla anterior
                    pos_anterior = (agente_pos_x, agente_pos_y)
                    if pos_anterior in ocupacion:
                        ocupacion[pos_anterior] -= 1
                        if ocupacion[pos_anterior] <= 0:
                            del ocupacion[pos_anterior]
                    
                    # Ocupar la nueva casilla (salvo si es la salida)
                    if siguiente_posicion != meta:
                        ocupacion[siguiente_posicion] = densidad_destino + 1
                    
                    caminos_recorridos[i].append(siguiente_posicion)
                    
                    # Costo de paso con penalización exponencial según densidad
                    if siguiente_posicion == meta or mapa[siguiente_pos_x, siguiente_pos_y] == -3:
                        costo_paso = 1
                    else:
                        costo_base = float(mapa[siguiente_pos_x, siguiente_pos_y])
                        costo_paso = round(float(costo_base * np.exp(densidad_destino)), 2)
                    
                    nuevo_costo = round(costo_acumulado + costo_paso, 2)
                    
                    if mapa_simulacion[siguiente_pos_x, siguiente_pos_y] == -1:
                        if siguiente_posicion in ocupacion:
                            ocupacion[siguiente_posicion] -= 1
                            if ocupacion[siguiente_posicion] <= 0:
                                del ocupacion[siguiente_posicion]
                        agentes_bfs[i] = (False, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                    elif siguiente_posicion == meta:
                        turnos_escape[i] = iteracion
                        agentes_bfs[i] = (True, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                    else:
                        agentes_bfs[i] = (agente[0], siguiente_pos_x, siguiente_pos_y, False, camino, nuevo_costo)
                else:
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_bfs[i] = (False, agente_pos_x, agente_pos_y, True, [], costo_acumulado)
        
        # Agentes atrapados sin camino
        agentes_vivos_sin_camino = []
        agentes_vivos_totales = []
        for i, ag in enumerate(agentes_bfs):
            if ag[0] and not ag[3]:
                agentes_vivos_totales.append(i)
                if len(ag[4]) == 0:
                    agentes_vivos_sin_camino.append(i)
        
        # Verificación de fuego estancado
        if not fuego_avanzo and len(agentes_vivos_sin_camino) == len(agentes_vivos_totales) and len(agentes_vivos_totales) > 0:
            fuego_puede_crecer = False
            filas_mapa, columnas_mapa = mapa_simulacion.shape
            posiciones_fuego = np.argwhere(mapa_simulacion == -1)
            movimientos_fuego = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for fuego_x, fuego_y in posiciones_fuego:
                for dx, dy in movimientos_fuego:
                    vx, vy = fuego_x + dx, fuego_y + dy
                    if 0 <= vx < filas_mapa and 0 <= vy < columnas_mapa and mapa_simulacion[vx, vy] > 0:
                        fuego_puede_crecer = True
                        break
                if fuego_puede_crecer:
                    break

            if not fuego_puede_crecer:
                for idx in agentes_vivos_sin_camino:
                    agente_atrapado = agentes_bfs[idx]
                    pos_atrapado = (agente_atrapado[1], agente_atrapado[2])
                    if pos_atrapado in ocupacion:
                        ocupacion[pos_atrapado] -= 1
                        if ocupacion[pos_atrapado] <= 0:
                            del ocupacion[pos_atrapado]
                    agentes_bfs[idx] = (False, agente_atrapado[1], agente_atrapado[2], True, [], agente_atrapado[5])
        
        # Condición de continuación
        quedan_agentes_activos = any(ag[0] and not ag[3] for ag in agentes_bfs)
        bucle_bfs = quedan_agentes_activos
    
    return calcular_metricas_simulacion(
        turnos_escape=turnos_escape,
        n_total=len(agentes_bfs),
        agentes=agentes_bfs,
        caminos_recorridos=caminos_recorridos,
        mapa_final=mapa_simulacion
    )


# Alias de compatibilidad
simular_profundidad_limitada = simular_amplitud


def ejecucion_2(num_repeticiones=80, cantidad_agentes=80, capacidad_maxima=CAPACIDAD_MAXIMA_ESTANDAR):
    """
    Ejecuta el experimento de 80 repeticiones para Búsqueda en Amplitud (BFS).
    Calcula y promedia las métricas de evacuación en cada mapa (Mapa 1, Mapa 2 y Mapa 3),
    midiendo y reportando el tiempo total transcurrido.
    """
    print("=" * 70)
    print("BÚSQUEDA EN AMPLITUD (BFS) - EXPERIMENTO DE 80 REPETICIONES")
    print(f"Cantidad de agentes: {cantidad_agentes}")
    print(f"Capacidad máxima por casilla: {capacidad_maxima}")
    print(f"Posición inicial del fuego: {POSICION_FUEGO_ESTANDAR}")
    print("=" * 70)
    
    tiempo_inicio = time.time()
    
    mapas = [
        ("Mapa 1 (Alta densidad / Cuello de botella)", mapa1),
        ("Mapa 2 (Densidad media / Laberinto corporativo)", mapa2),
        ("Mapa 3 (Baja densidad / Dispersión abierta)", mapa3)
    ]
    
    resultados_simulaciones = {}
    
    for nombre_mapa, mapa in mapas:
        print(f"\nIniciando evaluaciones en: {nombre_mapa} ({num_repeticiones} repeticiones)...")
        resultados_simulaciones[nombre_mapa] = []
        
        for rep in range(num_repeticiones):
            metricas = simular_amplitud(mapa, cantidad_agentes=cantidad_agentes, capacidad_maxima=capacidad_maxima)
            resultados_simulaciones[nombre_mapa].append(metricas)
            
            paso_progreso = max(1, num_repeticiones // 4)
            if (rep + 1) % paso_progreso == 0 or (rep + 1) == num_repeticiones:
                print(f"  -> Progreso: {rep + 1}/{num_repeticiones} repeticiones completadas.")
                
    return calcular_e_imprimir_metricas(
        resultados_simulaciones,
        num_repeticiones=num_repeticiones,
        cantidad_agentes=cantidad_agentes,
        tiempo_inicio=tiempo_inicio,
        nombre_algoritmo="Búsqueda en Amplitud (BFS)"
    )

ejecucion = ejecucion_2


def main():
    ejecucion_2()


if __name__ == "__main__":
    main()
