import numpy as np
import heapq
import random
import time
from funciones_auxiliar import (
    get_vecinos, 
    propagacion_de_incendio, 
    iniciar_fuego, 
    heuristica_manhattan, 
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

# Algoritmo de búsqueda A* (A-Star)
def a_star(grilla: np.ndarray, inicio, fin, heuristic=heuristica_manhattan):
    matriz = grilla
    largo, ancho = matriz.shape
    
    if inicio == fin:
        return 0, [inicio]
    
    # Inicializar la lista cerrada con celdas no visitadas
    closed_list = [[False for _ in range(ancho)] for _ in range(largo)]
    
    # Correspondencia de celdas con sus detalles (g, h, f, padre)
    cell_details = [[None for _ in range(ancho)] for _ in range(largo)]
    
    # Cola de prioridad para nodos abiertos
    open_list = []
    
    # Inicializar la celda de inicio
    h_inicio = heuristic(inicio, fin)
    cell_details[inicio[0]][inicio[1]] = (0, h_inicio, h_inicio, None)
    heapq.heappush(open_list, (cell_details[inicio[0]][inicio[1]][2], inicio))
    
    destino_encontrado = False
    
    while len(open_list) > 0:
        _, celda_actual = heapq.heappop(open_list)
        
        # Descartar nodos que ya fueron cerrados/expandidos previamente
        if closed_list[celda_actual[0]][celda_actual[1]]:
            continue
            
        closed_list[celda_actual[0]][celda_actual[1]] = True
        
        # Verificar si se ha alcanzado la celda de destino
        if celda_actual == fin:
            destino_encontrado = True
            break
        
        # Expandir vecinos de la celda actual y calcular sus costos
        for vecino, costo in get_vecinos(matriz, celda_actual):
            # Solo considerar vecinos que no estén en la lista cerrada
            if not closed_list[vecino[0]][vecino[1]]:
                # Calcular costos g, h y f para el vecino
                g_new = cell_details[celda_actual[0]][celda_actual[1]][0] + costo
                h_new = heuristic(vecino, fin)
                f_new = g_new + h_new
                
                # Actualizar los detalles de la celda si es un camino más corto o si aún no ha sido visitada
                if cell_details[vecino[0]][vecino[1]] is None or cell_details[vecino[0]][vecino[1]][2] > f_new:
                    cell_details[vecino[0]][vecino[1]] = (g_new, h_new, f_new, celda_actual)
                    heapq.heappush(open_list, (f_new, vecino))

    if not destino_encontrado:
        return None, None
    
    # Reconstruir camino desde el fin hacia el inicio
    camino = []
    actual = fin
    while actual is not None:
        camino.append(actual)
        actual = cell_details[actual[0]][actual[1]][3]
        
    camino.reverse()
    costo_total = cell_details[fin[0]][fin[1]][0]
    
    return costo_total, camino


def simular_a_star(mapa, cantidad_agentes=80, meta=None, capacidad_maxima=4, posicion_fuego=None):
    """
    Ejecuta la simulación de evacuación con Búsqueda A* (A-Star).
    - mapa: Matriz del entorno (mapa1, mapa2, mapa3, etc.)
    - cantidad_agentes: Puede ser un entero (número de agentes, por defecto 80 usando posiciones estándar)
      o una lista de coordenadas iniciales [(x, y), ...]
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
    
    agentes_a_star = [(True, pos[0], pos[1], False, [], 0) for pos in posiciones_iniciales]
    caminos_recorridos = [[(pos[0], pos[1])] for pos in posiciones_iniciales]
    turnos_escape = [None] * len(posiciones_iniciales)
    
    # Registro de ocupación de las casillas por agentes activos
    ocupacion = {}
    for pos in posiciones_iniciales:
        ocupacion[pos] = ocupacion.get(pos, 0) + 1
    
    bucle_a_star = True
    contador_propagacion = 0
    iteracion = 0
    max_iteraciones = 250
    
    while bucle_a_star:
        iteracion += 1
        if iteracion >= max_iteraciones:
            for i, agente in enumerate(agentes_a_star):
                if agente[0] and not agente[3]:
                    agentes_a_star[i] = (False, agente[1], agente[2], True, [], agente[5])
            break
        
        fuego_avanzo = False
        if contador_propagacion >= random.randint(2, 3):
            contador_propagacion = 0
            mapa_con_fuego_nuevo = propagacion_de_incendio(mapa_simulacion)
            fuego_avanzo = not np.array_equal(mapa_con_fuego_nuevo, mapa_simulacion)
            mapa_simulacion = mapa_con_fuego_nuevo
            
            for i, agente in enumerate(agentes_a_star):
                if agente[0] and not agente[3]:
                    agente_pos_x, agente_pos_y = agente[1], agente[2]
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_a_star[i] = (False, agente_pos_x, agente_pos_y, True, [], agente[5])
        else:
            contador_propagacion += 1
        
        cambio_matriz = mapa_simulacion_agentes is None or not np.array_equal(mapa_simulacion, mapa_simulacion_agentes)
        if cambio_matriz:
            mapa_simulacion_agentes = mapa_simulacion.copy()
            # Generar mapa con penalización exponencial y celdas saturadas bloqueadas
            mapa_con_congestion = generar_mapa_con_congestion(mapa_simulacion_agentes, ocupacion, meta, capacidad_maxima=capacidad_maxima)
            cache_caminos = {}
            for i, agente in enumerate(agentes_a_star):
                if agente[0] and not agente[3]:
                    pos_ag = (agente[1], agente[2])
                    if pos_ag not in cache_caminos:
                        costo_algoritmo, nuevo_camino = a_star(mapa_con_congestion, pos_ag, meta)
                        cache_caminos[pos_ag] = (costo_algoritmo, nuevo_camino)
                    else:
                        costo_algoritmo, nuevo_camino = cache_caminos[pos_ag]
                    if nuevo_camino is not None:
                        camino_pasos = nuevo_camino[1:]
                    else:
                        camino_pasos = []
                    
                    agentes_a_star[i] = (agente[0], agente[1], agente[2], agente[3], camino_pasos, agente[5])
        
        for i, agente in enumerate(agentes_a_star):
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
                        # Casilla no transitable por capacidad máxima alcanzada: el agente espera en su posición
                        camino.insert(0, siguiente_posicion)
                        continue
                    
                    # Actualizar ocupación: liberar casilla anterior
                    pos_anterior = (agente_pos_x, agente_pos_y)
                    if pos_anterior in ocupacion:
                        ocupacion[pos_anterior] -= 1
                        if ocupacion[pos_anterior] <= 0:
                            del ocupacion[pos_anterior]
                    
                    # Ocupar la nueva casilla (salvo si es la salida)
                    if siguiente_posicion != meta:
                        ocupacion[siguiente_posicion] = densidad_destino + 1
                    
                    caminos_recorridos[i].append(siguiente_posicion)
                    
                    # Costo de paso con penalización exponencial según densidad de ocupación
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
                        agentes_a_star[i] = (False, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                    elif siguiente_posicion == meta:
                        turnos_escape[i] = iteracion
                        agentes_a_star[i] = (True, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                    else:
                        agentes_a_star[i] = (agente[0], siguiente_pos_x, siguiente_pos_y, False, camino, nuevo_costo)
                else:
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_a_star[i] = (False, agente_pos_x, agente_pos_y, True, [], costo_acumulado)
        
        agentes_vivos_sin_camino = []
        for i in range(len(agentes_a_star)):
            agente = agentes_a_star[i]
            esta_vivo = agente[0]
            ha_terminado = agente[3]
            camino_pendiente = agente[4]
            if esta_vivo and not ha_terminado and len(camino_pendiente) == 0:
                agentes_vivos_sin_camino.append(i)
        
        agentes_vivos_totales = []
        for i in range(len(agentes_a_star)):
            agente = agentes_a_star[i]
            esta_vivo = agente[0]
            ha_terminado = agente[3]
            if esta_vivo and not ha_terminado:
                agentes_vivos_totales.append(i)
        
        if not fuego_avanzo and len(agentes_vivos_sin_camino) == len(agentes_vivos_totales) and len(agentes_vivos_totales) > 0:
            fuego_puede_crecer = False
            filas_mapa = mapa_simulacion.shape[0]
            columnas_mapa = mapa_simulacion.shape[1]
            posiciones_fuego = np.argwhere(mapa_simulacion == -1)
            movimientos_fuego = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for fuego_x, fuego_y in posiciones_fuego:
                for desplazamiento_x, desplazamiento_y in movimientos_fuego:
                    vecino_x = fuego_x + desplazamiento_x
                    vecino_y = fuego_y + desplazamiento_y
                    if 0 <= vecino_x < filas_mapa and 0 <= vecino_y < columnas_mapa:
                        if mapa_simulacion[vecino_x, vecino_y] > 0:
                            fuego_puede_crecer = True
                            break
                if fuego_puede_crecer:
                    break

            if not fuego_puede_crecer:
                for idx in agentes_vivos_sin_camino:
                    agente_atrapado = agentes_a_star[idx]
                    costo_acumulado_agente = agente_atrapado[5]
                    pos_atrapado = (agente_atrapado[1], agente_atrapado[2])
                    if pos_atrapado in ocupacion:
                        ocupacion[pos_atrapado] -= 1
                        if ocupacion[pos_atrapado] <= 0:
                            del ocupacion[pos_atrapado]
                    agentes_a_star[idx] = (False, agente_atrapado[1], agente_atrapado[2], True, [], costo_acumulado_agente)
        
        quedan_agentes_activos = False
        for agente in agentes_a_star:
            esta_vivo = agente[0]
            ha_terminado = agente[3]
            if esta_vivo and not ha_terminado:
                quedan_agentes_activos = True
                break
        
        bucle_a_star = quedan_agentes_activos
    
    return calcular_metricas_simulacion(
        turnos_escape=turnos_escape,
        n_total=len(agentes_a_star),
        agentes=agentes_a_star,
        caminos_recorridos=caminos_recorridos,
        mapa_final=mapa_simulacion
    )


def ejecucion_3(num_repeticiones=80, cantidad_agentes=80, capacidad_maxima=CAPACIDAD_MAXIMA_ESTANDAR):
    """
    Ejecuta el experimento de 80 repeticiones para Búsqueda A* (A-Star).
    Calcula y promedia las métricas de evacuación en cada mapa (Mapa 1, Mapa 2 y Mapa 3),
    midiendo y reportando el tiempo total transcurrido en horas, minutos y segundos.
    """
    print("=" * 70)
    print("BÚSQUEDA A* (A-STAR) - EXPERIMENTO DE 80 REPETICIONES")
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
            metricas = simular_a_star(mapa, cantidad_agentes=cantidad_agentes, capacidad_maxima=capacidad_maxima)
            resultados_simulaciones[nombre_mapa].append(metricas)
            
            paso_progreso = max(1, num_repeticiones // 4)
            if (rep + 1) % paso_progreso == 0 or (rep + 1) == num_repeticiones:
                print(f"  -> Progreso: {rep + 1}/{num_repeticiones} repeticiones completadas.")
                
    return calcular_e_imprimir_metricas(
        resultados_simulaciones,
        num_repeticiones=num_repeticiones,
        cantidad_agentes=cantidad_agentes,
        tiempo_inicio=tiempo_inicio,
        nombre_algoritmo="Búsqueda A* (A-Star)"
    )

# Alias genérico para facilitar llamadas
ejecucion = ejecucion_3


def main():
    ejecucion_3()


if __name__ == '__main__':
    main()
