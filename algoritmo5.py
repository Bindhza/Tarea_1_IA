import numpy as np
import random
import time
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

# Posibles movimientos en la grilla (arriba, abajo, izquierda, derecha)
movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def simular_camino(individuo, matriz, inicio, fin):
    """Simula el recorrido paso a paso de un individuo en la grilla."""
    camino = [inicio]
    actual = inicio
    
    for dx, dy in individuo:
        nuevo_x, nuevo_y = actual[0] + dx, actual[1] + dy
        if 0 <= nuevo_x < matriz.shape[0] and 0 <= nuevo_y < matriz.shape[1]:
            # Transitable si es camino (> 0) o salida (-3); muros (0) y fuego (-1) no transitables
            if matriz[nuevo_x, nuevo_y] > 0 or matriz[nuevo_x, nuevo_y] == -3:
                actual = (nuevo_x, nuevo_y)
                camino.append(actual)
                if actual == fin:
                    return camino, True
                    
    return camino, False


def calcular_fitness(individuo, matriz, inicio, fin):
    """Calcula el fitness premiando el avance hacia la meta, y penalizando congestión y bucles."""
    camino, llego = simular_camino(individuo, matriz, inicio, fin)
    pos_final = camino[-1]
    
    dist_final = abs(pos_final[0] - fin[0]) + abs(pos_final[1] - fin[1])
    dist_inicio = abs(inicio[0] - fin[0]) + abs(inicio[1] - fin[1])
    avance = dist_inicio - dist_final
    
    # Penalizar celdas con alto costo (congestión)
    costo_congestion = sum(matriz[x, y] for x, y in camino if matriz[x, y] > 0)
    # Penalizar bucles redundantes
    bucles = len(camino) - len(set(camino))
    
    if llego:
        return 10000.0 + (2000.0 / len(camino)) - (costo_congestion * 0.1)
    else:
        score = (100.0 / (dist_final + 1.0)) + max(0, avance * 5.0) - (bucles * 2.0) - (costo_congestion * 0.05)
        return max(0.01, score)


def create_poblacion_inicial(tamano_poblacion, longitud_camino, inicio, fin):
    """Genera la población inicial combinando exploración estocástica con sesgo hacia la meta."""
    poblacion = []
    
    dirs_preferidas = []
    if fin[0] > inicio[0]: dirs_preferidas.append((1, 0))
    elif fin[0] < inicio[0]: dirs_preferidas.append((-1, 0))
    if fin[1] > inicio[1]: dirs_preferidas.append((0, 1))
    elif fin[1] < inicio[1]: dirs_preferidas.append((0, -1))
    if not dirs_preferidas:
        dirs_preferidas = movimientos
        
    for _ in range(tamano_poblacion):
        individuo = []
        for _ in range(longitud_camino):
            if random.random() < 0.65:
                individuo.append(random.choice(dirs_preferidas))
            else:
                individuo.append(random.choice(movimientos))
        poblacion.append(individuo)
        
    return poblacion


def seleccion(poblacion, fitnesses, tamano_torneo=3):
    selected = []
    tamano_torneo = min(tamano_torneo, len(poblacion))
    poblacion_fitness = list(zip(poblacion, fitnesses))
    for _ in range(len(poblacion)):
        torneo = random.sample(poblacion_fitness, tamano_torneo)
        winner = max(torneo, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


def crossover(padre1, padre2):
    punto = random.randint(1, len(padre1) - 1)
    hijo1 = padre1[:punto] + padre2[punto:]
    hijo2 = padre2[:punto] + padre1[punto:]
    return hijo1, hijo2


def mutacion(individuo, tasa_mutacion, dirs_preferidas=None):
    individuo_mutado = list(individuo)
    for i in range(len(individuo_mutado)):
        if random.random() < tasa_mutacion:
            if dirs_preferidas and random.random() < 0.6:
                individuo_mutado[i] = random.choice(dirs_preferidas)
            else:
                individuo_mutado[i] = random.choice(movimientos)
    return individuo_mutado


def algoritmo_genetico(grilla, inicio, fin, poblacion_inicial=25, num_generaciones=25, tasa_mutacion=0.15):
    """
    Algoritmo genético adaptativo para navegación y evacuación.
    Si no encuentra el camino completo a la meta dentro de las generaciones dadas,
    retorna el camino del mejor individuo que más avanzó hacia la salida,
    evitando que los agentes queden congelados.
    """
    matriz = grilla
    largo, ancho = matriz.shape        

    if inicio == fin:
        return 0, [inicio]
    
    longitud_camino = min(120, (largo + ancho) * 2)
    
    dirs_preferidas = []
    if fin[0] > inicio[0]: dirs_preferidas.append((1, 0))
    elif fin[0] < inicio[0]: dirs_preferidas.append((-1, 0))
    if fin[1] > inicio[1]: dirs_preferidas.append((0, 1))
    elif fin[1] < inicio[1]: dirs_preferidas.append((0, -1))
    
    poblacion = create_poblacion_inicial(poblacion_inicial, longitud_camino, inicio, fin)
    
    mejor_global_camino = [inicio]
    mejor_global_dist = abs(inicio[0] - fin[0]) + abs(inicio[1] - fin[1])
    
    for generacion in range(num_generaciones):
        fitnesses = [calcular_fitness(ind, matriz, inicio, fin) for ind in poblacion]
        
        indice_mejor = max(range(len(poblacion)), key=lambda i: fitnesses[i])
        mejor_individuo = poblacion[indice_mejor]
        camino_mejor, llego = simular_camino(mejor_individuo, matriz, inicio, fin)
        
        if llego:
            costo_total = len(camino_mejor) - 1
            return costo_total, camino_mejor
        # 
        dist_actual = abs(camino_mejor[-1][0] - fin[0]) + abs(camino_mejor[-1][1] - fin[1])
        if dist_actual < mejor_global_dist or (dist_actual == mejor_global_dist and len(camino_mejor) > len(mejor_global_camino)):
            mejor_global_dist = dist_actual
            mejor_global_camino = camino_mejor
            
        padres = seleccion(poblacion, fitnesses)
        nueva_poblacion = [mejor_individuo]  # Elitismo
        
        while len(nueva_poblacion) < len(poblacion):
            padre1 = random.choice(padres)
            padre2 = random.choice(padres)
            
            hijo1, hijo2 = crossover(padre1, padre2)
            nueva_poblacion.append(mutacion(hijo1, tasa_mutacion, dirs_preferidas))
            if len(nueva_poblacion) < len(poblacion):
                nueva_poblacion.append(mutacion(hijo2, tasa_mutacion, dirs_preferidas))
                
        poblacion = nueva_poblacion
        
    # Si no alcanzó la meta completa, retornar el mejor avance para permitir desplazamiento continuo
    if len(mejor_global_camino) > 1:
        return len(mejor_global_camino) - 1, mejor_global_camino
        
    return None, None


def simular_algoritmo_genetico(mapa, cantidad_agentes=80, meta=None, capacidad_maxima=4, posicion_fuego=None):
    """
    Ejecuta la simulación de evacuación con Algoritmo Genético considerando capacidad máxima y congestión.
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
    
    agentes_genetico = [(True, pos[0], pos[1], False, [], 0) for pos in posiciones_iniciales]
    caminos_recorridos = [[(pos[0], pos[1])] for pos in posiciones_iniciales]
    turnos_escape = [None] * len(posiciones_iniciales)
    
    # Registro de ocupación de las casillas por agentes activos
    ocupacion = {}
    for pos in posiciones_iniciales:
        ocupacion[pos] = ocupacion.get(pos, 0) + 1
    
    bucle_genetico = True
    contador_propagacion = 0
    iteracion = 0
    max_iteraciones = 250
    
    while bucle_genetico:
        iteracion += 1
        if iteracion >= max_iteraciones:
            for i, agente in enumerate(agentes_genetico):
                if agente[0] and not agente[3]:
                    agentes_genetico[i] = (False, agente[1], agente[2], True, [], agente[5])
            break
        
        fuego_avanzo = False
        if contador_propagacion >= random.randint(2, 3):
            contador_propagacion = 0
            mapa_con_fuego_nuevo = propagacion_de_incendio(mapa_simulacion)
            fuego_avanzo = not np.array_equal(mapa_con_fuego_nuevo, mapa_simulacion)
            mapa_simulacion = mapa_con_fuego_nuevo
            
            # Verificar si algún agente fue alcanzado por el fuego
            for i, agente in enumerate(agentes_genetico):
                if agente[0] and not agente[3]:
                    agente_pos_x, agente_pos_y = agente[1], agente[2]
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        # Actualizar ocupación: liberar casilla del agente muerto
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_genetico[i] = (False, agente_pos_x, agente_pos_y, True, [], agente[5])
        else:
            contador_propagacion += 1
        
        cambio_matriz = mapa_simulacion_agentes is None or not np.array_equal(mapa_simulacion, mapa_simulacion_agentes)
        
        # Revisar si hay cambios en la matriz del mapa para recalcular caminos
        cache_caminos = {}
        if cambio_matriz:
            mapa_simulacion_agentes = mapa_simulacion.copy()
            # Generar mapa con penalización exponencial y celdas saturadas bloqueadas
            mapa_con_congestion = generar_mapa_con_congestion(mapa_simulacion_agentes, ocupacion, meta, capacidad_maxima=capacidad_maxima)
            for i, agente in enumerate(agentes_genetico):
                if agente[0] and not agente[3]:
                    pos_agente = (agente[1], agente[2])
                    if pos_agente not in cache_caminos:
                        _, nuevo_camino = algoritmo_genetico(mapa_con_congestion, pos_agente, meta)
                        cache_caminos[pos_agente] = nuevo_camino
                    else:
                        nuevo_camino = cache_caminos[pos_agente]
                    
                    if nuevo_camino is not None:
                        camino_pasos = list(nuevo_camino[1:])
                    else:
                        camino_pasos = []
                    
                    agentes_genetico[i] = (agente[0], agente[1], agente[2], agente[3], camino_pasos, agente[5])
        else:
            # Replanificar para agentes activos que ya consumieron su tramo de ruta previo
            for i, agente in enumerate(agentes_genetico):
                if agente[0] and not agente[3] and len(agente[4]) == 0:
                    pos_agente = (agente[1], agente[2])
                    if pos_agente not in cache_caminos:
                        mapa_con_congestion = generar_mapa_con_congestion(mapa_simulacion, ocupacion, meta, capacidad_maxima=capacidad_maxima)
                        _, nuevo_camino = algoritmo_genetico(mapa_con_congestion, pos_agente, meta)
                        cache_caminos[pos_agente] = nuevo_camino
                    else:
                        nuevo_camino = cache_caminos[pos_agente]
                    if nuevo_camino is not None:
                        agentes_genetico[i] = (agente[0], agente[1], agente[2], agente[3], list(nuevo_camino[1:]), agente[5])
        
        for i, agente in enumerate(agentes_genetico):
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
                        agentes_genetico[i] = (False, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                    elif siguiente_posicion == meta:
                        agentes_genetico[i] = (True, siguiente_pos_x, siguiente_pos_y, True, [], nuevo_costo)
                        turnos_escape[i] = iteracion
                    else:
                        agentes_genetico[i] = (agente[0], siguiente_pos_x, siguiente_pos_y, False, camino, nuevo_costo)
                else:
                    if mapa_simulacion[agente_pos_x, agente_pos_y] == -1:
                        pos_muerto = (agente_pos_x, agente_pos_y)
                        if pos_muerto in ocupacion:
                            ocupacion[pos_muerto] -= 1
                            if ocupacion[pos_muerto] <= 0:
                                del ocupacion[pos_muerto]
                        agentes_genetico[i] = (False, agente_pos_x, agente_pos_y, True, [], costo_acumulado)
        
        agentes_vivos_sin_camino = []
        for i in range(len(agentes_genetico)):
            agente = agentes_genetico[i]
            esta_vivo = agente[0]
            ha_terminado = agente[3]
            camino_pendiente = agente[4]
            if esta_vivo and not ha_terminado and len(camino_pendiente) == 0:
                agentes_vivos_sin_camino.append(i)
        
        agentes_vivos_totales = []
        for i in range(len(agentes_genetico)):
            agente = agentes_genetico[i]
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
                    agente_atrapado = agentes_genetico[idx]
                    costo_acumulado_agente = agente_atrapado[5]
                    pos_atrapado = (agente_atrapado[1], agente_atrapado[2])
                    if pos_atrapado in ocupacion:
                        ocupacion[pos_atrapado] -= 1
                        if ocupacion[pos_atrapado] <= 0:
                            del ocupacion[pos_atrapado]
                    agentes_genetico[idx] = (False, agente_atrapado[1], agente_atrapado[2], True, [], costo_acumulado_agente)
        
        quedan_agentes_activos = False
        for agente in agentes_genetico:
            esta_vivo = agente[0]
            ha_terminado = agente[3]
            if esta_vivo and not ha_terminado:
                quedan_agentes_activos = True
                break
        
        bucle_genetico = quedan_agentes_activos
    
    return calcular_metricas_simulacion(
        turnos_escape=turnos_escape,
        n_total=len(agentes_genetico),
        agentes=agentes_genetico,
        caminos_recorridos=caminos_recorridos,
        mapa_final=mapa_simulacion
    )


def ejecucion_5(num_repeticiones=80, cantidad_agentes=80, capacidad_maxima=CAPACIDAD_MAXIMA_ESTANDAR):
    """
    Ejecuta el experimento de 80 repeticiones para Algoritmo Genético.
    Calcula y promedia las métricas de evacuación en cada mapa (Mapa 1, Mapa 2 y Mapa 3),
    midiendo y reportando el tiempo total transcurrido en horas, minutos y segundos.
    """
    print("=" * 70)
    print("ALGORITMO GENÉTICO - EXPERIMENTO DE 80 REPETICIONES")
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
            metricas = simular_algoritmo_genetico(mapa, cantidad_agentes=cantidad_agentes, capacidad_maxima=capacidad_maxima)
            resultados_simulaciones[nombre_mapa].append(metricas)
            
            paso_progreso = max(1, num_repeticiones // 4)
            if (rep + 1) % paso_progreso == 0 or (rep + 1) == num_repeticiones:
                print(f"  -> Progreso: {rep + 1}/{num_repeticiones} repeticiones completadas.")
                
    return calcular_e_imprimir_metricas(
        resultados_simulaciones,
        num_repeticiones=num_repeticiones,
        cantidad_agentes=cantidad_agentes,
        tiempo_inicio=tiempo_inicio,
        nombre_algoritmo="Algoritmo Genético"
    )

# Alias genérico para facilitar llamadas
ejecucion = ejecucion_5


def main():
    ejecucion_5()


if __name__ == '__main__':
    main()