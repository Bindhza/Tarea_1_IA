# Tarea 1: Escape de la Torre - Inteligencia Artificial

## Nombre
- Benjamin Poblete

---

## Requisitos y Dependencias
La única dependencia externa requerida es `numpy`.

Para instalar las dependencias:
```bash
pip install numpy
```

---

## Reglas para Ejecutar el Código

> **Nota sobre persistencia:** Todos los resultados, datos del algoritmo y métricas de evacuación se imprimen en pantalla y **se guardan automáticamente en el archivo `resultados.txt`**.

El archivo principal del sistema es `implementacion.py`. Permite ejecutar el benchmarking controlado (80 repeticiones por defecto, configurable) en los 3 mapas de prueba para cada algoritmo.

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

## Algoritmos Implementados

1. **Búsqueda No Informada:**
   - **Búsqueda de Costo Uniforme (UCS)** (`algoritmo1.py`)
   - **Búsqueda en Amplitud (BFS)** (`algoritmo2.py`)

2. **Búsqueda Informada (Heurística Admisible):**
   - **Búsqueda A*** (`algoritmo3.py`)
   - **Greedy Best-First Search** (`algoritmo4.py`)

3. **Optimización Bioinspirada:**
   - **Algoritmo Genético Adaptativo** (`algoritmo5.py`)

---

## Entornos de Prueba (Mapas 50x50)

- **Mapa 1 (Alta densidad / Cuello de botella)** 
- **Mapa 2 (Densidad media / Laberinto corporativo)**
- **Mapa 3 (Baja densidad / Dispersión abierta)**

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

