# ============================================================
# PROYECTO INTEGRADOR: DINAMICA DEL NASDAQ
# ARCHIVO 03: CADENAS DE MARKOV (3 ESTADOS DE RIESGO)
#
# Objetivo:
# Clasificar cada mes del Nasdaq en un estado de riesgo y estimar
# la probabilidad de pasar de un estado a otro.
#
# Indicador: drawdown respecto al maximo de los ultimos 12 meses
#   DD_t = 1 - P_t / max(P_{t-11}, ..., P_t)
#
# Estados (criterio de la ficha del proyecto):
#   0 Normal       DD < 5%
#   1 Vulnerable   5% <= DD < 10%
#   2 Estres       DD >= 10%
#
# Por que maximo de 12 meses y no maximo historico:
# con el maximo historico, el Nasdaq habria estado en "Estres"
# de forma casi continua entre 2000 y 2015 (no recupero el pico
# de la burbuja puntocom). Eso mide una herida vieja, no el riesgo
# actual. La ventana de 12 meses coincide con el horizonte de
# monitoreo del Middle Office.
#
# Convencion de la matriz: filas = estado actual,
#                          columnas = estado del mes siguiente.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. RUTAS Y PARAMETROS

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_DATOS = BASE_DIR / "data" / "processed" / "base_exploracion.csv"
RUTA_TABLAS = BASE_DIR / "outputs" / "tablas"
RUTA_GRAFICOS = BASE_DIR / "outputs" / "graficos"
RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

VENTANA_MAXIMO = 12          # meses para el maximo movil
UMBRAL_VULNERABLE = 0.05     # 5% de drawdown
UMBRAL_ESTRES = 0.10         # 10% de drawdown
HORIZONTE = 12               # meses hacia adelante
ESTADOS = ["Normal", "Vulnerable", "Estres"]
N_ESTADOS = len(ESTADOS)


def clasificar_estado(drawdown):
    """Convierte un drawdown (0.07 = 7%) en estado 0, 1 o 2."""
    return np.select(
        [drawdown < UMBRAL_VULNERABLE, drawdown < UMBRAL_ESTRES],
        [0, 1],
        default=2,
    )


# 2. CARGA DE DATOS

datos = pd.read_csv(RUTA_DATOS, parse_dates=["Fecha"])
datos = datos.sort_values("Fecha").reset_index(drop=True)

# Verificar que los meses sean consecutivos (sin huecos)
ordinal = datos["Fecha"].dt.year * 12 + datos["Fecha"].dt.month
if not (ordinal.diff().dropna() == 1).all():
    raise ValueError("Hay meses faltantes: la cadena exige meses consecutivos.")

print("Observaciones:", len(datos))
print("Periodo:", datos["Fecha"].min().date(), "a", datos["Fecha"].max().date())


# 3. INDICADOR Y ESTADOS

maximo_12m = datos["NASDAQCOM"].rolling(VENTANA_MAXIMO, min_periods=1).max()
datos["Drawdown_12m"] = 1 - datos["NASDAQCOM"] / maximo_12m
datos["Estado"] = clasificar_estado(datos["Drawdown_12m"].to_numpy())
datos["Nombre_estado"] = datos["Estado"].map(dict(enumerate(ESTADOS)))

datos[["Fecha", "NASDAQCOM", "Drawdown_12m", "Estado", "Nombre_estado"]].to_csv(
    RUTA_TABLAS / "markov_secuencia_estados.csv", index=False
)

distribucion = (
    datos["Nombre_estado"].value_counts().reindex(ESTADOS).rename("Meses").to_frame()
)
distribucion["Proporcion"] = distribucion["Meses"] / distribucion["Meses"].sum()
distribucion.to_csv(RUTA_TABLAS / "markov_distribucion_estados.csv")
print("\nMeses por estado:\n", distribucion)


# 4. CONTEO DE TRANSICIONES Y MATRIZ

def matriz_transicion(estados):
    """Cuenta transiciones t -> t+1 y las convierte en probabilidades."""
    conteos = np.zeros((N_ESTADOS, N_ESTADOS))
    for actual, siguiente in zip(estados[:-1], estados[1:]):
        conteos[actual, siguiente] += 1
    probabilidades = conteos / conteos.sum(axis=1, keepdims=True)
    return conteos, probabilidades


conteos, P = matriz_transicion(datos["Estado"].to_numpy())

tabla_conteos = pd.DataFrame(conteos.astype(int), index=ESTADOS, columns=ESTADOS)
tabla_P = pd.DataFrame(P, index=ESTADOS, columns=ESTADOS)
tabla_conteos.to_csv(RUTA_TABLAS / "markov_conteos.csv")
tabla_P.to_csv(RUTA_TABLAS / "markov_matriz_transicion.csv")

print("\nConteos de transicion:\n", tabla_conteos)
print("\nMatriz de transicion:\n", tabla_P.round(3))
print("Suma de filas:", P.sum(axis=1).round(6))


# 5. PERMANENCIA Y DURACION ESPERADA
# Duracion esperada en un estado = 1 / (1 - p_ii)  (distribucion geometrica)

indicadores = pd.DataFrame({
    "Prob_permanencia": np.diag(P),
    "Prob_salida": 1 - np.diag(P),
    "Duracion_esperada_meses": 1 / (1 - np.diag(P)),
    "Transiciones_observadas": conteos.sum(axis=1).astype(int),
}, index=ESTADOS)
indicadores.to_csv(RUTA_TABLAS / "markov_indicadores.csv")
print("\nIndicadores:\n", indicadores.round(3))


# 6. DISTRIBUCION ESTACIONARIA (largo plazo)
# Vector pi tal que pi = pi P: autovector izquierdo con valor propio 1.

valores, vectores = np.linalg.eig(P.T)
pi = np.real(vectores[:, np.argmin(np.abs(valores - 1))])
pi = pi / pi.sum()
tabla_pi = pd.DataFrame({
    "Estacionaria": pi,
    "Observada": distribucion["Proporcion"].to_numpy(),
}, index=ESTADOS)
tabla_pi.to_csv(RUTA_TABLAS / "markov_distribucion_estacionaria.csv")
print("\nDistribucion estacionaria vs observada:\n", tabla_pi.round(3))


# 7. PROYECCION A 12 MESES DESDE EL ESTADO ACTUAL

estado_actual = int(datos["Estado"].iloc[-1])
fecha_actual = datos["Fecha"].iloc[-1]
print(f"\nEstado actual ({fecha_actual.date()}): {ESTADOS[estado_actual]}"
      f"  (drawdown 12m = {datos['Drawdown_12m'].iloc[-1]:.1%})")

# 7.1 Probabilidad de estar en cada estado en el mes h: fila de P^h
# 7.2 Probabilidad de haber tocado Estres al menos una vez hasta h:
#     se vuelve Estres "absorbente" y se toma P_abs^h
P_abs = P.copy()
P_abs[2] = [0, 0, 1]

filas = []
vector = np.eye(N_ESTADOS)[estado_actual]
vector_abs = vector.copy()
for h in range(1, HORIZONTE + 1):
    vector = vector @ P
    vector_abs = vector_abs @ P_abs
    filas.append({
        "Mes_horizonte": h,
        "Fecha": fecha_actual + pd.DateOffset(months=h),
        "P_Normal": vector[0],
        "P_Vulnerable": vector[1],
        "P_Estres": vector[2],
        "P_tocar_Estres_hasta_h": vector_abs[2],
    })
proyeccion = pd.DataFrame(filas)
proyeccion.to_csv(RUTA_TABLAS / "markov_proyeccion_12m.csv", index=False)
print("\nProyeccion desde el estado actual:\n",
      proyeccion.drop(columns="Fecha").round(3).to_string(index=False))


# 8. DIAGNOSTICOS DE LOS SUPUESTOS

# 8.1 Homogeneidad temporal: misma matriz en dos submuestras
mitad = len(datos) // 2
_, P1 = matriz_transicion(datos["Estado"].to_numpy()[:mitad + 1])
_, P2 = matriz_transicion(datos["Estado"].to_numpy()[mitad:])
fecha_corte = datos["Fecha"].iloc[mitad].date()
homogeneidad = pd.DataFrame({
    "Permanencia_1a_mitad": np.diag(P1),
    "Permanencia_2a_mitad": np.diag(P2),
}, index=ESTADOS)
homogeneidad.to_csv(RUTA_TABLAS / "markov_homogeneidad.csv")
print(f"\nHomogeneidad (corte en {fecha_corte}):\n", homogeneidad.round(3))

# 8.2 Calibracion historica: desde Normal, ¿con que frecuencia se toco
#     Estres en los 12 meses siguientes?  (comparar con P_abs^12)
estados = datos["Estado"].to_numpy()
toques = [
    (estados[t + 1:t + 1 + HORIZONTE] == 2).any()
    for t in range(len(estados) - HORIZONTE)
    if estados[t] == estado_actual
]
frecuencia_hist = float(np.mean(toques))
calibracion = pd.DataFrame({
    "Indicador": [
        "Estado_actual",
        "P_tocar_Estres_12m_Markov",
        "Frecuencia_historica_desde_estado_actual",
        "Casos_historicos",
    ],
    "Valor": [
        ESTADOS[estado_actual],
        proyeccion["P_tocar_Estres_hasta_h"].iloc[-1],
        frecuencia_hist,
        len(toques),
    ],
})
calibracion.to_csv(RUTA_TABLAS / "markov_calibracion.csv", index=False)
print("\nCalibracion:\n", calibracion.to_string(index=False))


# 9. GRAFICOS

colores = {0: "#2e7d32", 1: "#f9a825", 2: "#c62828"}

# 9.1 Nasdaq coloreado por estado
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(datos["Fecha"], datos["NASDAQCOM"], color="grey", lw=0.8)
for e in range(N_ESTADOS):
    m = datos["Estado"] == e
    ax1.scatter(datos.loc[m, "Fecha"], datos.loc[m, "NASDAQCOM"],
                s=9, color=colores[e], label=ESTADOS[e])
ax1.set_yscale("log")
ax1.set_ylabel("Nasdaq (escala log)")
ax1.legend()
ax1.set_title("Estados de riesgo del Nasdaq (drawdown vs maximo de 12 meses)")
ax2.fill_between(datos["Fecha"], -100 * datos["Drawdown_12m"], color="#c62828", alpha=0.4)
ax2.axhline(-100 * UMBRAL_VULNERABLE, ls="--", color=colores[1], lw=1)
ax2.axhline(-100 * UMBRAL_ESTRES, ls="--", color=colores[2], lw=1)
ax2.set_ylabel("Drawdown 12m (%)")
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RUTA_GRAFICOS / "markov_estados.png", dpi=150)
plt.close(fig)

# 9.2 Matriz de transicion
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(N_ESTADOS), ESTADOS)
ax.set_yticks(range(N_ESTADOS), ESTADOS)
ax.set_xlabel("Estado del mes siguiente")
ax.set_ylabel("Estado actual")
for i in range(N_ESTADOS):
    for j in range(N_ESTADOS):
        ax.text(j, i, f"{P[i, j]:.2f}\n(n={int(conteos[i, j])})", ha="center",
                va="center", color="white" if P[i, j] > 0.5 else "black")
fig.colorbar(im, ax=ax)
ax.set_title("Matriz de transicion mensual")
fig.tight_layout()
fig.savefig(RUTA_GRAFICOS / "markov_matriz.png", dpi=150)
plt.close(fig)

# 9.3 Proyeccion de probabilidades
fig, ax = plt.subplots(figsize=(10, 5))
for e, col in zip(range(N_ESTADOS), ["P_Normal", "P_Vulnerable", "P_Estres"]):
    ax.plot(proyeccion["Mes_horizonte"], proyeccion[col], marker="o",
            color=colores[e], label=ESTADOS[e])
ax.plot(proyeccion["Mes_horizonte"], proyeccion["P_tocar_Estres_hasta_h"], ls="--",
        color=colores[2], label="Tocar Estres al menos una vez")
ax.set_xlabel("Meses hacia adelante")
ax.set_ylabel("Probabilidad")
ax.set_title(f"Proyeccion Markov desde el estado actual ({ESTADOS[estado_actual]})")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(RUTA_GRAFICOS / "markov_proyeccion.png", dpi=150)
plt.close(fig)

print("\nMODULO 03 COMPLETADO")
