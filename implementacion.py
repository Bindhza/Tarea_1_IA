import sys
from algoritmo1 import ejecucion_1
from algoritmo2 import ejecucion_2, simular_amplitud
from algoritmo3 import ejecucion_3
from algoritmo4 import ejecucion_4
from algoritmo5 import ejecucion_5
from funciones_auxiliar import (
    POSICION_FUEGO_ESTANDAR,
    CAPACIDAD_MAXIMA_ESTANDAR
)

def main():
    print("=" * 70)
    print("SIMULACIÓN DE EVACUACIÓN - TAREA 1 IA (Entorno 50x50 - 80 Agentes)")
    print(f"Cantidad estándar de agentes:  80 (pisos superiores)")
    print(f"Capacidad máxima por casilla:  {CAPACIDAD_MAXIMA_ESTANDAR} agentes")
    print(f"Posición inicial del fuego:    {POSICION_FUEGO_ESTANDAR}")
    print("Archivo de resultados:         resultados.txt (persistente)")
    print("=" * 70)
    
    opcion = sys.argv[1] if len(sys.argv) > 1 else None
    repeticiones = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    
    if opcion is None:
        print("\nSelecciona el algoritmo a evaluar:")
        print("  1. Búsqueda de Costo Uniforme (UCS)")
        print("  2. Búsqueda en Amplitud (BFS)")
        print("  3. Búsqueda A* (A-Star)")
        print("  4. Greedy Best-First Search")
        print("  5. Algoritmo Genético")
        print("  6. Evaluar TODOS los algoritmos secuencialmente")
        try:
            opcion = input("\nIngresa una opción (1-6) [por defecto 1]: ").strip()
            if not opcion:
                opcion = "1"
        except (EOFError, KeyboardInterrupt):
            opcion = "1"
            
    print(f"\nIniciando benchmarking con {repeticiones} repeticiones por mapa...")
    
    if opcion == "1":
        ejecucion_1(num_repeticiones=repeticiones, cantidad_agentes=80)
    elif opcion == "2":
        ejecucion_2(num_repeticiones=repeticiones, cantidad_agentes=80)
    elif opcion == "3":
        ejecucion_3(num_repeticiones=repeticiones, cantidad_agentes=80)
    elif opcion == "4":
        ejecucion_4(num_repeticiones=repeticiones, cantidad_agentes=80)
    elif opcion == "5":
        ejecucion_5(num_repeticiones=repeticiones, cantidad_agentes=80)
    elif opcion in ("6", "all", "todos"):
        print("\n>>> EVALUANDO 1/5: BÚSQUEDA DE COSTO UNIFORME (UCS)")
        ejecucion_1(num_repeticiones=repeticiones, cantidad_agentes=80)
        print("\n>>> EVALUANDO 2/5: BÚSQUEDA EN AMPLITUD (BFS)")
        ejecucion_2(num_repeticiones=repeticiones, cantidad_agentes=80)
        print("\n>>> EVALUANDO 3/5: BÚSQUEDA A* (A-STAR)")
        ejecucion_3(num_repeticiones=repeticiones, cantidad_agentes=80)
        print("\n>>> EVALUANDO 4/5: GREEDY BEST-FIRST SEARCH")
        ejecucion_4(num_repeticiones=repeticiones, cantidad_agentes=80)
        print("\n>>> EVALUANDO 5/5: ALGORITMO GENÉTICO")
        ejecucion_5(num_repeticiones=repeticiones, cantidad_agentes=80)
    else:
        print(f"Opción no reconocida ({opcion}). Ejecutando Búsqueda de Costo Uniforme (UCS) por defecto.")
        ejecucion_1(num_repeticiones=repeticiones, cantidad_agentes=80)

if __name__ == "__main__":
    main()
