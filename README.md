# Tarea 1: Escape de la Torre - Inteligencia Artificial

Simulación de evacuación y navegación multiagente ante un incendio dinámico en un edificio de múltiples niveles, evaluando el desempeño de algoritmos bajo tres paradigmas de búsqueda y optimización.

---

## Integrantes
- Integrante 1: [Nombre y Apellido] - [RUT / Correo]
- Integrante 2: [Nombre y Apellido] - [RUT / Correo]
- Integrante 3: [Nombre y Apellido] - [RUT / Correo]

---

## Requisitos y Dependencias
El proyecto está implementado en **Python 3**. La única dependencia externa requerida es `numpy`.

Para instalar las dependencias:
```bash
pip install numpy
```

---

## Reglas para Ejecutar el Código

> **Nota sobre persistencia:** Todos los resultados, datos del algoritmo y métricas de evacuación se imprimen en pantalla y **se guardan automáticamente en `resultados.txt`**, asegurando que la información de cada ejecución quede guardada de forma persistente para la confección del informe.

El punto de entrada principal del sistema es `implementacion.py`. Permite ejecutar el benchmarking controlado (80 repeticiones por defecto, configurable) en los 3 mapas de prueba para cada algoritmo.

### Modo Interactivo (Menú):
```bash
python3 implementacion.py
```
Desplegará un menú numérico para seleccionar qué algoritmo evaluar o si se desean evaluar todos de manera secuencial.

### Modo por Línea de Comandos:
Puedes especificar directamente el número de algoritmo (1 a 6) y opcionalmente el número de repeticiones (por ejemplo, 80 o 200):
```bash
# Ejecutar Costo Uniforme (UCS) con 80 repeticiones
python3 implementacion.py 1 80

# Ejecutar BFS con 80 repeticiones
python3 implementacion.py 2 80

# Ejecutar A* con 80 repeticiones
python3 implementacion.py 3 80

# Ejecutar Greedy Best-First con 80 repeticiones
python3 implementacion.py 4 80

# Ejecutar Algoritmo Genético con 80 repeticiones
python3 implementacion.py 5 80

# Ejecutar TODOS los algoritmos secuencialmente (benchmark completo)
python3 implementacion.py all 80
```

---

## Paradigmas y Algoritmos Implementados

1. **Búsqueda No Informada:**
   - **Búsqueda de Costo Uniforme (UCS)** (`algoritmo1.py`): Explora expandiendo el nodo con menor costo acumulado $g(n)$, optimizado con descarte en $O(1)$ de entradas obsoletas en la cola de prioridad.
   - **Búsqueda en Amplitud (BFS)** (`algoritmo2.py`): Explora sistemáticamente por niveles utilizando una cola FIFO (`collections.deque`), garantizando encontrar el camino con la menor cantidad de pasos/transiciones ortogonales hacia la salida.

2. **Búsqueda Informada (Heurística Admisible):**
   - **Búsqueda A*** (`algoritmo3.py`): Minimiza $f(n) = g(n) + h(n)$ garantizando rutas de costo mínimo bajo la heurística admisible de distancia Manhattan ortogonal.
   - **Greedy Best-First Search** (`algoritmo4.py`): Búsqueda voraz guiada exclusivamente por la heurística admisible de Manhattan $h(n)$ hacia la salida.

3. **Optimización Bioinspirada:**
   - **Algoritmo Genético Adaptativo** (`algoritmo5.py`): Algoritmo genético con selección por torneo, cruce en un punto, mutación y elitismo. Implementa planificación continua por horizonte deslizante, retornando el mejor avance hacia la salida para evitar que los agentes se queden congelados.

---

## Entornos de Prueba (Mapas 50x50)

- **Mapa 1 (Alta densidad / Cuello de botella):** Pasillos angostos con convergencia forzada hacia un único pasaje central hacia la salida, generando alta saturación.
- **Mapa 2 (Densidad media / Laberinto corporativo):** Conjunto de salas conectadas por intersecciones y cruces con densidad intermedia.
- **Mapa 3 (Baja densidad / Dispersión abierta):** Entorno abierto con columnas y tabiques aislados que permiten múltiples rutas alternativas.

---

## Estructura de la Matriz

- ` 0`: Muralla / Obstáculo (intransitable)
- `-1`: Fuego activo (intransitable e irreversible)
- ` 1`: Camino transitable (costo base penalizado dinámicamente por congestión)
- `-3`: Única salida de evacuación común

---

## Fuentes
- Búsqueda de Costo Uniforme : https://www.geeksforgeeks.org/artificial-intelligence/uniform-cost-search-ucs-in-ai/
- Búsqueda en Amplitud (BFS) : https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
- A* : https://www.geeksforgeeks.org/dsa/a-search-algorithm/
- Greedy Best First Search : https://www.geeksforgeeks.org/dsa/greedy-best-first-search-algorithm/
- Algoritmo Genético: https://www.datacamp.com/es/tutorial/genetic-algorithm-python

*la ia generativa ha sido utilizada para temas conceptuales, explicaciones, ayuda a la lectura de códigos extraídos de internet o facilitar la documentación, es decir, como apoyo del aprendizaje, pero en ningún caso fue usada para crear código*
