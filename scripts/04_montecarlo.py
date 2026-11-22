# ============================================================
# PROYECTO INTEGRADOR: DINAMICA DEL NASDAQ
# ARCHIVO 04: SIMULACION MONTE CARLO BASADA EN EL VAR(2)
#
# Objetivo:
# Simular 5.000 trayectorias de 12 meses de las 4 variables con
# la dinamica del VAR(2) y convertir cada trayectoria del Nasdaq
# en estados de riesgo (mismo criterio del modulo 03).
#
# Mecanismo generador:
#   y_t = c + A1 y_{t-1} + A2 y_{t-2} + e_t
#   e_t = vector de residuos historicos del VAR tomado al azar
#         (bootstrap de filas completas).
#
# Por que bootstrap y no una normal:
# la prueba de normalidad del modulo 02 rechaza normalidad
# (colas pesadas). Tomar la fila completa de residuos conserva
# la correlacion contemporanea entre las 4 variables
# (p.ej. cuando cae el Nasdaq suele subir el diferencial Baa).
#
# Integracion con los otros modulos:
#   VAR (02)    -> mecanismo que genera las trayectorias
#   Markov (03) -> criterio de estados para clasificarlas
#   Monte Carlo -> probabilidad de Estres e incertidumbre
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR


# 1. RUTAS Y PARAMETROS

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_TRANSFORMADA = BASE_DIR / "data" / "processed" / "base_transformada.csv"
RUTA_NIVELES = BASE_DIR / "data" / "processed" / "base_exploracion.csv"
RUTA_TABLAS = BASE_DIR / "outputs" / "tablas"
RUTA_GRAFICOS = BASE_DIR / "outputs" / "graficos"
RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

VARIABLES = ["NASDAQ_RET", "WTI_RET", "BAA10YM", "GS10_DIF"]  # igual que en 02
REZAGOS = 2                  # VAR(2) elegido en el modulo 02
N_SIMULACIONES = 5_000
HORIZONTE = 12
SEMILLA = 12345

# Mismo criterio de estados del modulo 03
VENTANA_MAXIMO = 12
UMBRAL_VULNERABLE = 0.05
UMBRAL_ESTRES = 0.10
ESTADOS = ["Normal", "Vulnerable", "Estres"]


# 2. DATOS Y ESTIMACION DEL VAR(2)
# Se re-estima con la misma especificacion del modulo 02
# (mismas variables, mismos rezagos, constante).

datos = pd.read_csv(RUTA_TRANSFORMADA, parse_dates=["Fecha"]).set_index("Fecha")
datos.index = pd.DatetimeIndex(datos.index, freq="MS")
datos = datos[VARIABLES]

niveles = pd.read_csv(RUTA_NIVELES, parse_dates=["Fecha"]).set_index("Fecha")

modelo = VAR(datos).fit(REZAGOS, trend="c")
print(modelo.summary().__str__()[:600], "...\n")

constante = modelo.intercept            # (k,)
coeficientes = modelo.coefs             # (p, k, k): A1, A2
residuos = modelo.resid.to_numpy()      # (T-p, k)
residuos = residuos - residuos.mean(axis=0)  # centrar (media ~0 por construccion)

print("Residuos disponibles para el bootstrap:", len(residuos))


# 3. SIMULACION

rng = np.random.default_rng(SEMILLA)
k = len(VARIABLES)

# Ultimos 2 meses observados como punto de partida
ultimos = datos.to_numpy()[-REZAGOS:]                 # (p, k)
simulado = np.zeros((N_SIMULACIONES, HORIZONTE, k))

for s in range(N_SIMULACIONES):
    historia = list(ultimos)                           # [y_{t-2}, y_{t-1}]
    choques = residuos[rng.integers(0, len(residuos), HORIZONTE)]
    for h in range(HORIZONTE):
        y = constante.copy()
        for rezago in range(REZAGOS):
            y += coeficientes[rezago] @ historia[-1 - rezago]
        y += choques[h]
        simulado[s, h] = y
        historia.append(y)

# 4. DEL RENDIMIENTO SIMULADO AL NIVEL DEL NASDAQ
# NASDAQ_RET es rentabilidad logaritmica x100, por eso se compone con exp:
#   P_{t+h} = P_t * exp( sum(r) / 100 )

precio_actual = niveles["NASDAQCOM"].iloc[-1]
ret_nasdaq = simulado[:, :, 0]
indice = precio_actual * np.exp(np.cumsum(ret_nasdaq, axis=1) / 100)

# Drawdown vs maximo de 12 meses: la ventana mezcla los ultimos 11 meses
# reales con los meses simulados.
historia_precios = niveles["NASDAQCOM"].to_numpy()[-(VENTANA_MAXIMO - 1):]
camino = np.hstack([np.tile(historia_precios, (N_SIMULACIONES, 1)), indice])
drawdown = np.zeros((N_SIMULACIONES, HORIZONTE))
for h in range(HORIZONTE):
    ventana = camino[:, h + 1:h + 1 + VENTANA_MAXIMO]
    drawdown[:, h] = 1 - ventana[:, -1] / ventana.max(axis=1)

estados = np.select([drawdown < UMBRAL_VULNERABLE, drawdown < UMBRAL_ESTRES],
                    [0, 1], default=2)
fechas = pd.date_range(datos.index[-1] + pd.offsets.MonthBegin(1),
                       periods=HORIZONTE, freq="MS")


# 5. RESULTADOS POR HORIZONTE

toco_estres = np.maximum.accumulate(estados == 2, axis=1)
ret_acum = (np.exp(np.cumsum(ret_nasdaq, axis=1) / 100) - 1) * 100   # % simple

resumen = pd.DataFrame({
    "Fecha": fechas,
    "Mes_horizonte": np.arange(1, HORIZONTE + 1),
    "P_Normal": (estados == 0).mean(axis=0),
    "P_Vulnerable": (estados == 1).mean(axis=0),
    "P_Estres": (estados == 2).mean(axis=0),
    "P_tocar_Estres_hasta_h": toco_estres.mean(axis=0),
    "Indice_p5": np.percentile(indice, 5, axis=0),
    "Indice_p50": np.percentile(indice, 50, axis=0),
    "Indice_p95": np.percentile(indice, 95, axis=0),
    "Ret_acum_p5_pct": np.percentile(ret_acum, 5, axis=0),
    "Ret_acum_p50_pct": np.percentile(ret_acum, 50, axis=0),
    "Ret_acum_p95_pct": np.percentile(ret_acum, 95, axis=0),
})
resumen.to_csv(RUTA_TABLAS / "mc_resumen_horizontes.csv", index=False)
print(resumen.drop(columns="Fecha").round(3).to_string(index=False))


# 6. ESCENARIOS A 12 MESES Y RIESGO DE COLA

final = ret_acum[:, -1]
var95 = -np.percentile(final, 5)
cvar95 = -final[final <= np.percentile(final, 5)].mean()

escenarios = pd.DataFrame({
    "Escenario": ["Adverso (p5)", "Central (mediana)", "Favorable (p95)"],
    "Rentabilidad_12m_pct": np.percentile(final, [5, 50, 95]),
    "Nivel_Nasdaq_12m": np.percentile(indice[:, -1], [5, 50, 95]),
})
escenarios.to_csv(RUTA_TABLAS / "mc_escenarios_12m.csv", index=False)

metricas = pd.DataFrame({
    "Indicador": [
        "Simulaciones", "Horizonte_meses", "Semilla", "Nasdaq_inicial",
        "Rentabilidad_media_12m_pct", "Prob_rentabilidad_negativa_12m",
        "VaR95_12m_pct", "CVaR95_12m_pct",
        "P_Estres_mes12", "P_tocar_Estres_12m",
        "P_drawdown_max_mayor_20pct_12m",
    ],
    "Valor": [
        N_SIMULACIONES, HORIZONTE, SEMILLA, precio_actual,
        final.mean(), (final < 0).mean(),
        var95, cvar95,
        resumen["P_Estres"].iloc[-1], resumen["P_tocar_Estres_hasta_h"].iloc[-1],
        (drawdown.max(axis=1) > 0.20).mean(),
    ],
})
metricas.to_csv(RUTA_TABLAS / "mc_metricas.csv", index=False)
print("\n", escenarios.round(2).to_string(index=False))
print("\n", metricas.to_string(index=False))


# 7. ESTABILIDAD NUMERICA
# ¿La probabilidad clave cambia si usamos menos simulaciones?

tamanos = [250, 500, 1000, 2000, 3000, 4000, 5000]
estabilidad = pd.DataFrame({
    "Simulaciones": tamanos,
    "P_tocar_Estres_12m": [toco_estres[:n, -1].mean() for n in tamanos],
})
p = estabilidad["P_tocar_Estres_12m"]
estabilidad["Error_estandar"] = np.sqrt(p * (1 - p) / estabilidad["Simulaciones"])
estabilidad.to_csv(RUTA_TABLAS / "mc_estabilidad.csv", index=False)
print("\n", estabilidad.round(4).to_string(index=False))


# 8. TRAYECTORIAS PARA LA APP (indice y drawdown maximo)
# Se guarda solo lo necesario para que la app recalcule probabilidades
# con el umbral que elija el usuario.

tray = pd.DataFrame(indice, columns=[f"M{h}" for h in range(1, HORIZONTE + 1)])
tray["DD_max_12m"] = drawdown.max(axis=1)
tray.round(4).to_csv(RUTA_TABLAS / "mc_trayectorias.csv.gz", index=False,
                     compression="gzip")


# 9. GRAFICOS

fig, ax = plt.subplots(figsize=(11, 5))
hist = niveles["NASDAQCOM"].iloc[-36:]
ax.plot(hist.index, hist, color="black", lw=1.5, label="Observado")
for s in range(100):
    ax.plot(fechas, indice[s], color="#1565c0", alpha=0.08, lw=0.8)
ax.fill_between(fechas, resumen["Indice_p5"], resumen["Indice_p95"],
                color="#1565c0", alpha=0.15, label="Banda 5%-95%")
ax.plot(fechas, resumen["Indice_p50"], color="#1565c0", lw=2, label="Mediana")
ax.set_title(f"Nasdaq: {N_SIMULACIONES:,} trayectorias simuladas con el VAR(2)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RUTA_GRAFICOS / "mc_trayectorias.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(final, bins=80, color="#1565c0", alpha=0.75)
for valor, nombre in zip(np.percentile(final, [5, 50, 95]), ["p5", "p50", "p95"]):
    ax.axvline(valor, ls="--", color="black")
    ax.text(valor, ax.get_ylim()[1] * 0.92, f" {nombre}: {valor:.1f}%")
ax.set_xlabel("Rentabilidad acumulada a 12 meses (%)")
ax.set_title("Distribucion simulada de la rentabilidad del Nasdaq a 12 meses")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RUTA_GRAFICOS / "mc_distribucion_12m.png", dpi=150)
plt.close(fig)

print("\nMODULO 04 COMPLETADO")
