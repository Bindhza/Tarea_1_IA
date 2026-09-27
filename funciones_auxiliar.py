import random
import numpy as np
import time

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
    movimientos = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    nueva_matriz = np.copy(matriz)
    
    # Recorrer la matriz para encontrar celdas en llamas
    for i, j in np.ndindex(matriz.shape):
        if matriz[i, j] == -1:
            vecinos_incendiables = []
            for dx, dy in movimientos:
                nuevo_x, nuevo_y = i + dx, j + dy
                if 0 <= nuevo_x < matriz.shape[0] and 0 <= nuevo_y < matriz.shape[1] and matriz[nuevo_x, nuevo_y] > 0:
                    vecinos_incendiables.append((nuevo_x, nuevo_y)) 
            
            num_a_quemar = min(random.randint(2, 4), len(vecinos_incendiables))
            if num_a_quemar > 0:
                elegidos = random.sample(vecinos_incendiables, num_a_quemar)
                for nx, ny in elegidos:
                    nueva_matriz[nx, ny] = -1
                    
    return nueva_matriz

# Posiciones iniciales estándar y convenientes para 80 agentes (distribuidas en los pisos superiores)
# Coordenadas transitables y deterministas en los 3 mapas 50x50
POSICIONES_AGENTES_ESTANDAR = [
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11),
    (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 19), (1, 20), (1, 21), (1, 22), (1, 23),
    (1, 24), (1, 25), (1, 26), (1, 27), (1, 28), (1, 29), (1, 31), (1, 32), (1, 33), (1, 34),
    (1, 35), (1, 37), (1, 38), (1, 39), (1, 40), (1, 41), (1, 43), (1, 44), (1, 45), (1, 46),
    (1, 47), (1, 48), (2, 8), (2, 10), (2, 14), (2, 27), (2, 37), (2, 38), (2, 44), (3, 2),
    (3, 32), (3, 37), (4, 2), (4, 10), (4, 26), (4, 32), (4, 37), (5, 1), (5, 2), (5, 3),
    (5, 4), (5, 5), (5, 7), (5, 9), (5, 10), (5, 11), (5, 13), (5, 14), (5, 15), (5, 16),
    (5, 17), (5, 19), (5, 20), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25), (5, 26), (5, 27)
]

# Posición inicial estándar y conveniente para el fuego (zona intermedia del edificio)
# Amenaza el eje central hacia la salida sin bloquearla de inmediato, forzando interacción con congestión y desvíos
POSICION_FUEGO_ESTANDAR = (15, 20)

# Capacidad máxima estándar por celda recomendada para simulaciones con 80 agentes
CAPACIDAD_MAXIMA_ESTANDAR = 6


def obtener_posiciones_estandar(mapa, cantidad_agentes=80):
    """
    Retorna posiciones iniciales convenientes, no aleatorias y deterministas para los agentes.
    Distribuye los agentes a lo largo de las casillas transitables de los pisos superiores,
    garantizando pruebas estandarizadas y reproducibles en los 3 mapas 50x50.
    """
    if cantidad_agentes <= len(POSICIONES_AGENTES_ESTANDAR):
        return list(POSICIONES_AGENTES_ESTANDAR[:cantidad_agentes])
    
    posiciones = list(POSICIONES_AGENTES_ESTANDAR)
    for r in range(1, mapa.shape[0] - 5):
        for c in range(1, mapa.shape[1] - 1):
            coord = (r, c)
            if mapa[r, c] > 0 and coord != POSICION_FUEGO_ESTANDAR and coord not in posiciones:
                posiciones.append(coord)
                if len(posiciones) == cantidad_agentes:
                    return posiciones
    return posiciones

def iniciar_fuego(matriz, posiciones_excluidas=None, posicion_fuego=None):
    """
    Inicia el fuego en la matriz en una posición conveniente y determinista.
    - matriz: Matriz del mapa
    - posiciones_excluidas: Coordenadas que no pueden iniciar con fuego (e.g. agentes o salida)
    - posicion_fuego: Coordenada fija deseada. Si es None, utiliza POSICION_FUEGO_ESTANDAR (15, 20).
    Retorna una nueva matriz con el fuego iniciado.
    """
    nueva_matriz = np.copy(matriz)
    posiciones_transitables = np.argwhere(nueva_matriz > 0)
    
    if posiciones_excluidas is not None:
        posiciones_validas = [
            tuple(posicion) for posicion in posiciones_transitables 
            if tuple(posicion) not in posiciones_excluidas
        ]
    else:
        posiciones_validas = [tuple(posicion) for posicion in posiciones_transitables]
    
    if len(posiciones_validas) == 0:
        raise ValueError("No hay posiciones transitables para iniciar el fuego.")
    
    if posicion_fuego is not None:
        coord_fuego = tuple(posicion_fuego)
        if coord_fuego not in posiciones_validas:
            raise ValueError(f"La posición de fuego especificada {coord_fuego} no es transitable o está excluida.")
    else:
        # Usar la posición conveniente predeterminada estándar
        if POSICION_FUEGO_ESTANDAR in posiciones_validas:
            coord_fuego = POSICION_FUEGO_ESTANDAR
        else:
            coord_fuego = posiciones_validas[0]
            
    nueva_matriz[coord_fuego[0], coord_fuego[1]] = -1
    return nueva_matriz

def posiciones_iniciales_aleatorias(mapa, cantidad_agentes=4):
    """
    Genera posiciones iniciales aleatorias para los agentes en el mapa (modo aleatorio opcional).
    - mapa: Matriz del entorno (mapa1, mapa2, mapa3, etc.)
    - cantidad_agentes: Número de agentes a colocar (por defecto 4)
    Retorna una lista de coordenadas iniciales [(x, y), ...]
    """
    posiciones_transitables = np.argwhere(mapa > 0)
    if len(posiciones_transitables) < cantidad_agentes:
        raise ValueError("No hay suficientes posiciones transitables para colocar a todos los agentes.")
    posiciones_aleatorias = random.sample(list(map(tuple, posiciones_transitables)), cantidad_agentes)
    return posiciones_aleatorias

def heuristica_manhattan(a, b):
    """Calcula la distancia Manhattan entre dos coordenadas a y b."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def calcular_costo_penalizado(costo_base, ocupacion, capacidad_maxima=4):
    """
    Calcula el costo de atravesar una casilla con penalización exponencial según la ocupación.
    - Si la ocupación alcanza o supera la capacidad máxima (>= capacidad_maxima agentes), no es transitable (retorna 0).
    - Si la ocupación es menor, el costo base se incrementa de forma exponencial:
      costo = costo_base * exp(ocupacion).
    """
    if ocupacion >= capacidad_maxima:
        return 0  # 0 indica no transitable (obstáculo / muro)
    return round(float(costo_base * np.exp(ocupacion)), 2)

def generar_mapa_con_congestion(mapa_base, ocupacion, meta, capacidad_maxima=4):
    """
    Genera una copia del mapa aplicando las penalizaciones exponenciales por congestión
    y marcando como no transitables (0) las casillas que alcanzaron la capacidad física máxima.
    """
    mapa_dinamico = mapa_base.astype(float).copy()
    for (x, y), densidad in ocupacion.items():
        if (x, y) != meta and mapa_base[x, y] > 0:
            if densidad >= capacidad_maxima:
                mapa_dinamico[x, y] = 0  # No transitable si alcanza la capacidad máxima
            else:
                mapa_dinamico[x, y] = calcular_costo_penalizado(mapa_base[x, y], densidad, capacidad_maxima)
    return mapa_dinamico
    
def calcular_metricas_simulacion(turnos_escape, n_total, agentes=None, caminos_recorridos=None, mapa_final=None):
    """
    Calcula las métricas de rendimiento de una simulación individual a partir de los turnos de escape.
    - turnos_escape: Lista de turnos en que cada agente alcanzó la salida (None si no sobrevivió).
    - n_total: Cantidad total inicial de agentes.
    - agentes: Lista opcional con el estado final de los agentes.
    - caminos_recorridos: Lista opcional con la trayectoria de cada agente.
    - mapa_final: Matriz del mapa con el fuego final.
    Retorna un diccionario con las métricas requeridas.
    """
    sobrevivientes_tiempos = [t for t in turnos_escape if t is not None]
    n_sobrevivientes = len(sobrevivientes_tiempos)
    tasa_supervivencia = round((n_sobrevivientes / n_total) * 100.0, 2) if n_total > 0 else 0.0

    if n_sobrevivientes > 0:
        tiempo_despeje = max(sobrevivientes_tiempos)
        media_tiempo = round(float(np.mean(sobrevivientes_tiempos)), 2)
        std_tiempo = round(float(np.std(sobrevivientes_tiempos)), 2)
        min_tiempo = int(min(sobrevivientes_tiempos))
        max_tiempo = int(max(sobrevivientes_tiempos))
    else:
        tiempo_despeje = 0
        media_tiempo = 0.0
        std_tiempo = 0.0
        min_tiempo = 0
        max_tiempo = 0

    return {
        'tasa_supervivencia': tasa_supervivencia,
        'tiempo_despeje': tiempo_despeje,
        'media_tiempo': media_tiempo,
        'std_tiempo': std_tiempo,
        'desviacion_estandar_tiempo': std_tiempo,
        'min_tiempo': min_tiempo,
        'max_tiempo': max_tiempo,
        'n_sobrevivientes': n_sobrevivientes,
        'n_total': n_total,
        'turnos_escape': turnos_escape,
        'agentes': agentes,
        'caminos_recorridos': caminos_recorridos,
        'mapa_final': mapa_final
    }

def calcular_e_imprimir_metricas(resultados_por_mapa, num_repeticiones, cantidad_agentes, tiempo_inicio, nombre_algoritmo="Algoritmo", archivo_resultados="resultados.txt"):
    """
    Calcula los promedios y estadísticos descriptivos de las métricas obtenidas durante
    la ejecución de múltiples repeticiones de simulación, imprime el resumen final en pantalla
    y lo almacena de forma persistente en un archivo de texto (resultados.txt).
    
    - resultados_por_mapa: Diccionario {nombre_mapa: lista_de_metricas_o_dict_acumulado}
    - num_repeticiones: Número de repeticiones ejecutadas por mapa
    - cantidad_agentes: Número total de agentes iniciales
    - tiempo_inicio: Timestamp time.time() del inicio de la ejecución
    - nombre_algoritmo: Nombre del algoritmo evaluado
    - archivo_resultados: Nombre del archivo de texto para persistencia de resultados
    
    Retorna un diccionario con los promedios calculados y la duración formateada.
    """
    resultados_globales = {}
    
    for nombre_mapa, datos_mapa in resultados_por_mapa.items():
        if isinstance(datos_mapa, list):
            metricas_acumuladas = {
                'tasa_supervivencia': [m['tasa_supervivencia'] for m in datos_mapa],
                'tiempo_despeje': [m['tiempo_despeje'] for m in datos_mapa],
                'media_tiempo': [m['media_tiempo'] for m in datos_mapa],
                'std_tiempo': [m['std_tiempo'] for m in datos_mapa],
                'min_tiempo': [m['min_tiempo'] for m in datos_mapa],
                'max_tiempo': [m['max_tiempo'] for m in datos_mapa],
                'n_sobrevivientes': [m['n_sobrevivientes'] for m in datos_mapa]
            }
        elif isinstance(datos_mapa, dict):
            metricas_acumuladas = datos_mapa
        else:
            raise ValueError(f"Formato de datos no soportado para el mapa {nombre_mapa}")

        tiempos_despeje_validos = [t for t in metricas_acumuladas['tiempo_despeje'] if t > 0]
        if len(tiempos_despeje_validos) > 0:
            media_despeje = round(float(np.mean(tiempos_despeje_validos)), 2)
            std_despeje = round(float(np.std(tiempos_despeje_validos)), 2)
            min_despeje = int(np.min(tiempos_despeje_validos))
            max_despeje = int(np.max(tiempos_despeje_validos))
        else:
            media_despeje = 0.0
            std_despeje = 0.0
            min_despeje = 0
            max_despeje = 0

        promedios = {
            'promedio_tasa_supervivencia': round(float(np.mean(metricas_acumuladas['tasa_supervivencia'])), 2),
            'std_tasa_supervivencia': round(float(np.std(metricas_acumuladas['tasa_supervivencia'])), 2),
            'promedio_sobrevivientes': round(float(np.mean(metricas_acumuladas['n_sobrevivientes'])), 2),
            'media_tiempo_despeje': media_despeje,
            'std_tiempo_despeje': std_despeje,
            'min_tiempo_despeje': min_despeje,
            'max_tiempo_despeje': max_despeje,
            'promedio_media_tiempo': round(float(np.mean(metricas_acumuladas['media_tiempo'])), 2),
            'promedio_std_tiempo': round(float(np.mean(metricas_acumuladas['std_tiempo'])), 2)
        }
        
        resultados_globales[nombre_mapa] = {
            'promedios': promedios,
            'historico': metricas_acumuladas
        }
        
    tiempo_fin = time.time()
    duracion_total = tiempo_fin - tiempo_inicio
    
    horas = int(duracion_total // 3600)
    minutos = int((duracion_total % 3600) // 60)
    segundos = round(duracion_total % 60, 2)
    tiempo_str = f"{horas} horas, {minutos} minutos, {segundos} segundos (Total: {round(duracion_total, 2)} s)"
    
    bloque = []
    bloque.append("=" * 70)
    bloque.append(f"RESUMEN DE BENCHMARKING ({num_repeticiones} REPETICIONES) - {nombre_algoritmo}")
    bloque.append(f"Fecha y hora de registro:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    bloque.append(f"Cantidad total de agentes:   {cantidad_agentes}")
    bloque.append("=" * 70)
    for nombre_mapa, datos in resultados_globales.items():
        p = datos['promedios']
        bloque.append(f"\n[{nombre_mapa}]")
        bloque.append("  • Tasa de supervivencia:")
        bloque.append(f"      - Promedio:                            {p['promedio_tasa_supervivencia']}% ({p['promedio_sobrevivientes']}/{cantidad_agentes} agentes)")
        bloque.append(f"      - Desviación estándar (supervivencia): {p['std_tasa_supervivencia']}%")
        bloque.append("  • Distribución del tiempo total de despeje (turnos):")
        bloque.append(f"      - Media:                               {p['media_tiempo_despeje']} turnos")
        bloque.append(f"      - Desviación estándar (tiempo):        {p['std_tiempo_despeje']} turnos")
        bloque.append(f"      - Valor mínimo:                        {p['min_tiempo_despeje']} turnos")
        bloque.append(f"      - Valor máximo:                        {p['max_tiempo_despeje']} turnos")
        
    bloque.append("\n" + "-" * 70)
    bloque.append(f"Tiempo total transcurrido ({nombre_algoritmo}): {tiempo_str}")
    bloque.append("-" * 70)
    
    texto_resumen = "\n".join(bloque)
    
    # 1. Mostrar resumen en terminal
    print("\n" + texto_resumen)
    
    # 2. Guardar de forma persistente en archivo de texto (resultados.txt)
    try:
        with open(archivo_resultados, "a", encoding="utf-8") as f_txt:
            f_txt.write(texto_resumen + "\n\n")
        print(f"\n[INFO] Resultados guardados de forma persistente en '{archivo_resultados}'")
    except Exception as e:
        print(f"\n[ADVERTENCIA] No se pudo guardar en '{archivo_resultados}': {e}")
        
    return {
        'resultados': resultados_globales,
        'tiempo_total_segundos': duracion_total,
        'tiempo_formateado': tiempo_str
    }

# Mapa 1: Alta densidad de obstáculos / Cuello de botella (50x50)
# Pasillos angostos que convergen hacia un cuello de botella central que lleva a la salida (-3) en (49, 25)
mapa1 = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
])

# Mapa 2: Densidad media / Laberinto corporativo (50x50)
# Múltiples salas conectadas por intersecciones centrales y pasillos con cruces
mapa2 = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  1,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0,  0,  1,  1,  0],
    [ 0,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
])

# Mapa 3: Baja densidad / Dispersión abierta (50x50)
# Entorno semiabierto con columnas y tabiques aislados, permitiendo múltiples rutas alternativas
mapa3 = np.array([
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  0,  0,  1,  1,  1,  0,  0,  0,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0]
])
