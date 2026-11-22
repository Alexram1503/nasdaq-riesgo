# ============================================================
# PROYECTO INTEGRADOR NASDAQ
# MODULO 02: MODELO VAR
# Version revisada: comparacion con muestra comun e intervalos
# ============================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import InterpolationWarning


# ============================================================
# 1. RUTAS Y CONFIGURACION
# ============================================================

RUTA_BASE = (
    Path(__file__).resolve().parent.parent  # raiz del repositorio
    / "data" / "processed" / "base_transformada.csv"
)

CARPETA_PROYECTO = RUTA_BASE.parents[2]
RUTA_TABLAS = CARPETA_PROYECTO / "outputs" / "tablas"
RUTA_GRAFICOS = CARPETA_PROYECTO / "outputs" / "graficos"

RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

VARIABLES = ["NASDAQ_RET", "WTI_RET", "BAA10YM", "GS10_DIF"]
HORIZONTE = 12
ALPHA = 0.05
REZAGO_FINAL = 2


# ============================================================
# 2. CARGA Y PREPARACION DE DATOS
# ============================================================

print("\n======================================")
print("MODULO 02: MODELO VAR")
print("======================================")

if not RUTA_BASE.exists():
    raise FileNotFoundError(
        f"No se encontro la base transformada: {RUTA_BASE}"
    )

datos = pd.read_csv(RUTA_BASE, sep=None, engine="python")

print("\nColumnas encontradas:")
print(datos.columns.tolist())

columna_fecha = next(
    (
        columna for columna in datos.columns
        if columna.strip().lower() in ["fecha", "date", "observation_date"]
    ),
    None
)

if columna_fecha is None:
    raise ValueError(
        "No se encontro una columna de fecha en la base transformada."
    )

datos[columna_fecha] = pd.to_datetime(
    datos[columna_fecha],
    format="%Y-%m-%d",
    errors="raise"
)

datos = datos.sort_values(columna_fecha).set_index(columna_fecha)
datos.index = pd.DatetimeIndex(datos.index, freq="MS")

faltantes = [variable for variable in VARIABLES if variable not in datos.columns]
if faltantes:
    raise ValueError(
        f"Faltan variables necesarias: {faltantes}. "
        f"Columnas disponibles: {datos.columns.tolist()}"
    )

datos_modelo = datos[VARIABLES].copy()
datos_modelo = datos_modelo.apply(pd.to_numeric, errors="coerce")

if datos_modelo.isna().any().any():
    filas_con_na = int(datos_modelo.isna().any(axis=1).sum())
    print(
        f"\nADVERTENCIA: se eliminaran {filas_con_na} filas "
        "con valores faltantes o no numericos."
    )
    datos_modelo = datos_modelo.dropna()

datos_modelo = datos_modelo.sort_index()

if datos_modelo.index.has_duplicates:
    raise ValueError("La base contiene fechas duplicadas.")

if not datos_modelo.index.is_monotonic_increasing:
    raise ValueError("Las fechas no estan ordenadas cronologicamente.")

if len(datos_modelo) <= 30:
    raise ValueError("No hay suficientes observaciones para estimar el VAR.")

print("\nPrimeras observaciones:")
print(datos_modelo.head())

print("\nDimensiones:")
print(datos_modelo.shape)

print("\nPeriodo:")
print(datos_modelo.index.min(), "a", datos_modelo.index.max())

print("\nFrecuencia del indice:")
print(datos_modelo.index.freq)

print("\n¿Fechas ordenadas?:")
print(datos_modelo.index.is_monotonic_increasing)

print("\n¿Fechas duplicadas?:")
print(datos_modelo.index.duplicated().sum())

print("\nNumero de observaciones:")
print(len(datos_modelo))

datos_modelo.describe().to_csv(
    RUTA_TABLAS / "estadisticas_descriptivas_VAR.csv"
)


# ============================================================
# 3. PRUEBAS DE ESTACIONARIEDAD
# ============================================================

print("\n======================================")
print("PRUEBAS DE ESTACIONARIEDAD")
print("======================================")

resultados_estacionariedad = []

for variable in VARIABLES:
    serie = datos_modelo[variable]

    adf = adfuller(serie, autolag="AIC")

    # KPSS puede emitir InterpolationWarning cuando el p-valor real
    # esta por encima del maximo tabulado. Se conserva el p-valor
    # reportado y se deja una nota interpretativa en la salida.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        kpss_resultado = kpss(
            serie,
            regression="c",
            nlags="auto"
        )

    resultados_estacionariedad.append({
        "Variable": variable,
        "ADF estadistico": adf[0],
        "ADF p-valor": adf[1],
        "KPSS estadistico": kpss_resultado[0],
        "KPSS p-valor reportado": kpss_resultado[1]
    })

tabla_estacionariedad = pd.DataFrame(resultados_estacionariedad)

print(tabla_estacionariedad)
print(
    "\nNota KPSS: cuando el p-valor reportado es 0.1, "
    "puede indicar que el valor real es superior al limite de la tabla."
)

tabla_estacionariedad.to_csv(
    RUTA_TABLAS / "pruebas_estacionariedad.csv",
    index=False
)


# ============================================================
# 4. SELECCION DE REZAGOS
# ============================================================

print("\n======================================")
print("SELECCION DE REZAGOS")
print("======================================")

modelo_seleccion = VAR(datos_modelo)
seleccion = modelo_seleccion.select_order(maxlags=12)

print(seleccion.summary())

rezagos = seleccion.selected_orders
print("\nRezagos sugeridos:")
print(rezagos)

pd.DataFrame({
    "Criterio": ["AIC", "BIC", "HQIC", "FPE"],
    "Rezagos": [
        rezagos["aic"],
        rezagos["bic"],
        rezagos["hqic"],
        rezagos["fpe"]
    ]
}).to_csv(
    RUTA_TABLAS / "seleccion_rezagos.csv",
    index=False
)


# ============================================================
# 5. COMPARACION VAR(1) Y VAR(2) CON LA MISMA MUESTRA
# ============================================================

print("\n======================================")
print("COMPARACION VAR(1) Y VAR(2)")
print("======================================")

# El VAR(2) estimado sobre la base completa utiliza como variable
# dependiente las observaciones desde la tercera fila.
# Para que VAR(1) utilice exactamente esas mismas fechas dependientes,
# su base de entrada comienza en la segunda fila original.
modelo_var1_comun = VAR(datos_modelo.iloc[1:]).fit(1, trend="c")
modelo_var2_comun = VAR(datos_modelo).fit(2, trend="c")

if modelo_var1_comun.nobs != modelo_var2_comun.nobs:
    raise RuntimeError(
        "La comparacion VAR(1)-VAR(2) no quedo con el mismo numero "
        "de observaciones. Revisar la preparacion de la muestra."
    )

comparacion = pd.DataFrame({
    "Modelo": ["VAR(1)", "VAR(2)"],
    "Observaciones": [
        modelo_var1_comun.nobs,
        modelo_var2_comun.nobs
    ],
    "AIC": [
        modelo_var1_comun.aic,
        modelo_var2_comun.aic
    ],
    "BIC": [
        modelo_var1_comun.bic,
        modelo_var2_comun.bic
    ],
    "HQIC": [
        modelo_var1_comun.hqic,
        modelo_var2_comun.hqic
    ],
    "FPE": [
        modelo_var1_comun.fpe,
        modelo_var2_comun.fpe
    ]
})

print("\nComparacion calculada sobre la misma muestra efectiva:")
print(comparacion)

comparacion.to_csv(
    RUTA_TABLAS / "comparacion_VAR1_VAR2.csv",
    index=False
)


# ============================================================
# 6. ESTIMACION DEL MODELO FINAL VAR(2)
# ============================================================

print("\n======================================")
print("ESTIMACION DEL MODELO FINAL VAR(2)")
print("======================================")

modelo_final = VAR(datos_modelo).fit(REZAGO_FINAL, trend="c")

print(modelo_final.summary())

with open(
    RUTA_TABLAS / "resumen_modelo_VAR2.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write(str(modelo_final.summary()))

modelo_final.params.to_csv(
    RUTA_TABLAS / "coeficientes_VAR2.csv"
)

modelo_final.pvalues.to_csv(
    RUTA_TABLAS / "pvalores_VAR2.csv"
)


# ============================================================
# 7. ESTABILIDAD DEL MODELO
# ============================================================

print("\n======================================")
print("ESTABILIDAD DEL MODELO")
print("======================================")

es_estable = modelo_final.is_stable()
raices = np.asarray(modelo_final.roots)

print("¿Modelo estable?:", es_estable)
print("\nRaices caracteristicas:")
print(raices)
print("\nModulos de las raices:")
print(np.abs(raices))

pd.DataFrame({
    "Raiz": raices,
    "Modulo de la raiz": np.abs(raices)
}).to_csv(
    RUTA_TABLAS / "raices_VAR2.csv",
    index=False
)


# ============================================================
# 8. AUTOCORRELACION DE LOS RESIDUOS
# ============================================================

print("\n======================================")
print("AUTOCORRELACION DE RESIDUOS")
print("======================================")

prueba_autocorrelacion = modelo_final.test_whiteness(
    nlags=12,
    signif=ALPHA,
    adjusted=True
)

print(prueba_autocorrelacion)

with open(
    RUTA_TABLAS / "prueba_autocorrelacion.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write(str(prueba_autocorrelacion))


# ============================================================
# 9. NORMALIDAD DE LOS RESIDUOS
# ============================================================

print("\n======================================")
print("NORMALIDAD DE LOS RESIDUOS")
print("======================================")

prueba_normalidad = modelo_final.test_normality(signif=ALPHA)
print(prueba_normalidad)

with open(
    RUTA_TABLAS / "prueba_normalidad.txt",
    "w",
    encoding="utf-8"
) as archivo:
    archivo.write(str(prueba_normalidad))


# ============================================================
# 10. DIAGNOSTICO GRAFICO DE RESIDUOS
# ============================================================

print("\n======================================")
print("DIAGNOSTICO GRAFICO DE RESIDUOS")
print("======================================")

residuos = modelo_final.resid
residuos.to_csv(RUTA_TABLAS / "residuos_VAR2.csv")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for i, variable in enumerate(VARIABLES):
    ax = axes.flatten()[i]
    ax.plot(residuos.index, residuos[variable])
    ax.axhline(0, color="red", linestyle="--")
    ax.set_title(f"Residuos de {variable}")
    ax.set_xlabel("Fecha")

plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "residuos_temporales_VAR2.png",
    dpi=300
)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for i, variable in enumerate(VARIABLES):
    sns.histplot(
        residuos[variable],
        kde=True,
        ax=axes.flatten()[i]
    )
    axes.flatten()[i].set_title(f"Distribucion de {variable}")

plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "histogramas_residuos_VAR2.png",
    dpi=300
)
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for i, variable in enumerate(VARIABLES):
    sm.qqplot(
        residuos[variable],
        line="s",
        ax=axes.flatten()[i]
    )
    axes.flatten()[i].set_title(f"Q-Q Plot de {variable}")

plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "qqplots_residuos_VAR2.png",
    dpi=300
)
plt.close()


# ============================================================
# 11. HETEROCEDASTICIDAD ARCH
# ============================================================

print("\n======================================")
print("PRUEBAS ARCH")
print("======================================")

resultados_arch = []

for variable in VARIABLES:
    resultado = het_arch(residuos[variable], nlags=12)

    resultados_arch.append({
        "Variable": variable,
        "Estadistico ARCH": resultado[0],
        "p-valor ARCH": resultado[1],
        "Rechaza homocedasticidad 5%": resultado[1] < ALPHA
    })

tabla_arch = pd.DataFrame(resultados_arch)

print(tabla_arch)

tabla_arch.to_csv(
    RUTA_TABLAS / "pruebas_ARCH_VAR2.csv",
    index=False
)


# ============================================================
# 12. CAUSALIDAD DE GRANGER
# ============================================================

print("\n======================================")
print("CAUSALIDAD DE GRANGER")
print("======================================")

resultados_granger = []

for causa in VARIABLES:
    for efecto in VARIABLES:
        if causa != efecto:
            prueba = modelo_final.test_causality(
                caused=efecto,
                causing=[causa],
                kind="f"
            )

            resultados_granger.append({
                "Causa": causa,
                "Efecto": efecto,
                "Estadistico F": prueba.test_statistic,
                "p-valor": prueba.pvalue,
                "Significativo 5%": prueba.pvalue < ALPHA
            })

tabla_granger = pd.DataFrame(resultados_granger)

print(tabla_granger)

tabla_granger.to_csv(
    RUTA_TABLAS / "causalidad_Granger_VAR2.csv",
    index=False
)


# ============================================================
# 13. FUNCIONES IMPULSO-RESPUESTA (IRF)
# ============================================================

print("\n======================================")
print("FUNCIONES IMPULSO-RESPUESTA")
print("======================================")

irf = modelo_final.irf(HORIZONTE)

# Grafico de respuestas impulso-respuesta.
# orth=False conserva la forma reducida; no impone una ordenacion
# recursiva de Cholesky para interpretar los shocks.
fig_irf = irf.plot(
    orth=False,
    signif=ALPHA,
    figsize=(15, 12)
)

plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "funciones_IRF_VAR2.png",
    dpi=300
)
plt.close()

# Guardar respuestas puntuales en formato largo y con nombres.
valores_irf = irf.irfs
filas_irf = []

for horizonte_irf in range(valores_irf.shape[0]):
    for i, respuesta in enumerate(VARIABLES):
        for j, shock in enumerate(VARIABLES):
            filas_irf.append({
                "Horizonte": horizonte_irf,
                "Variable_respuesta": respuesta,
                "Variable_shock": shock,
                "Respuesta_IRF": valores_irf[horizonte_irf, i, j]
            })

tabla_irf = pd.DataFrame(filas_irf)
tabla_irf.to_csv(
    RUTA_TABLAS / "valores_IRF_VAR2.csv",
    index=False
)

# Bandas de confianza Monte Carlo para las IRF.
# Se fija semilla para facilitar la reproducibilidad.
try:
    irf_inferior, irf_superior = irf.errband_mc(
        orth=False,
        repl=1000,
        signif=ALPHA,
        seed=12345
    )

    filas_bandas_irf = []

    for h in range(valores_irf.shape[0]):
        for i, respuesta in enumerate(VARIABLES):
            for j, shock in enumerate(VARIABLES):
                filas_bandas_irf.append({
                    "Horizonte": h,
                    "Variable_respuesta": respuesta,
                    "Variable_shock": shock,
                    "Respuesta_IRF": valores_irf[h, i, j],
                    "Limite_inferior_95": irf_inferior[h, i, j],
                    "Limite_superior_95": irf_superior[h, i, j]
                })

    pd.DataFrame(filas_bandas_irf).to_csv(
        RUTA_TABLAS / "intervalos_IRF_VAR2.csv",
        index=False
    )

    print(
        "IRF calculadas y guardadas con bandas Monte Carlo "
        "del 95%."
    )

except Exception as error:
    print(
        "ADVERTENCIA: no fue posible calcular las bandas Monte Carlo "
        f"de las IRF: {error}"
    )
    print("Las respuestas puntuales y el grafico IRF si fueron guardados.")


# ============================================================
# 14. DESCOMPOSICION DE VARIANZA (FEVD)
# ============================================================

print("\n======================================")
print("DESCOMPOSICION DE VARIANZA")
print("======================================")

fevd = modelo_final.fevd(HORIZONTE)
print(fevd.summary())

fevd.plot(figsize=(14, 10))
plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "descomposicion_FEVD_VAR2.png",
    dpi=300
)
plt.close()

fevd_valores = fevd.decomp
filas_fevd = []

for i, variable in enumerate(VARIABLES):
    for horizonte_fevd in range(fevd_valores.shape[1]):
        for j, shock in enumerate(VARIABLES):
            filas_fevd.append({
                "Variable": variable,
                "Horizonte": horizonte_fevd + 1,
                "Shock": shock,
                "Proporcion": fevd_valores[i, horizonte_fevd, j]
            })

tabla_fevd = pd.DataFrame(filas_fevd)

tabla_fevd.to_csv(
    RUTA_TABLAS / "valores_FEVD_VAR2.csv",
    index=False
)


# ============================================================
# 15. PRONOSTICOS A 12 MESES CON INTERVALOS
# ============================================================

print("\n======================================")
print("PRONOSTICOS A 12 MESES")
print("======================================")

valores_finales = datos_modelo.values[-modelo_final.k_ar:]

pronosticos, limite_inferior, limite_superior = (
    modelo_final.forecast_interval(
        y=valores_finales,
        steps=HORIZONTE,
        alpha=ALPHA
    )
)

fechas_futuras = pd.date_range(
    start=datos_modelo.index[-1] + pd.offsets.MonthBegin(1),
    periods=HORIZONTE,
    freq="MS"
)

tabla_pronosticos = pd.DataFrame(
    pronosticos,
    index=fechas_futuras,
    columns=VARIABLES
)

tabla_limites_inferiores = pd.DataFrame(
    limite_inferior,
    index=fechas_futuras,
    columns=VARIABLES
)

tabla_limites_superiores = pd.DataFrame(
    limite_superior,
    index=fechas_futuras,
    columns=VARIABLES
)

tabla_pronosticos.to_csv(
    RUTA_TABLAS / "pronosticos_VAR2_12_meses.csv"
)

tabla_intervalos = pd.DataFrame(index=fechas_futuras)

for variable in VARIABLES:
    tabla_intervalos[f"{variable}_Pronostico"] = tabla_pronosticos[variable]
    tabla_intervalos[f"{variable}_Inferior_95"] = (
        tabla_limites_inferiores[variable]
    )
    tabla_intervalos[f"{variable}_Superior_95"] = (
        tabla_limites_superiores[variable]
    )

print("\nPronosticos puntuales:")
print(tabla_pronosticos)

print("\nPronosticos con intervalos de confianza del 95%:")
print(tabla_intervalos)

tabla_intervalos.to_csv(
    RUTA_TABLAS / "pronosticos_VAR2_intervalos_95.csv"
)


# ============================================================
# 16. GRAFICOS DE PRONOSTICOS CON INTERVALOS
# ============================================================

print("\n======================================")
print("GRAFICOS DE PRONOSTICOS")
print("======================================")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

for i, variable in enumerate(VARIABLES):
    ax = axes.flatten()[i]

    ax.plot(
        datos_modelo.index[-48:],
        datos_modelo[variable].tail(48),
        label="Historico"
    )

    ax.plot(
        tabla_pronosticos.index,
        tabla_pronosticos[variable],
        linestyle="--",
        label="Pronostico"
    )

    ax.fill_between(
        tabla_pronosticos.index,
        tabla_limites_inferiores[variable].values,
        tabla_limites_superiores[variable].values,
        alpha=0.20,
        label="Intervalo de confianza 95%"
    )

    ax.axvline(
        datos_modelo.index[-1],
        color="red",
        linestyle=":"
    )

    ax.set_title(f"Pronostico de {variable}")
    ax.set_xlabel("Fecha")
    ax.legend()

plt.tight_layout()
plt.savefig(
    RUTA_GRAFICOS / "pronosticos_VAR2_intervalos_95.png",
    dpi=300
)
plt.close()


# ============================================================
# 17. RESUMEN DE ARCHIVOS GENERADOS
# ============================================================

print("\n======================================")
print("MODULO 02 EJECUTADO")
print("======================================")

print("\nTablas guardadas en:")
print(RUTA_TABLAS)

print("\nGraficos guardados en:")
print(RUTA_GRAFICOS)

print("\nModelo estimado: VAR(2)")
print(f"Horizonte de pronostico: {HORIZONTE} meses")
print(f"Observaciones del modelo final: {modelo_final.nobs}")
print(f"Modelo estable: {es_estable}")

print("\nArchivos principales nuevos o actualizados:")
print("- comparacion_VAR1_VAR2.csv")
print("- valores_IRF_VAR2.csv")
print("- intervalos_IRF_VAR2.csv (si el calculo Monte Carlo se completa)")
print("- pronosticos_VAR2_intervalos_95.csv")
print("- pronosticos_VAR2_intervalos_95.png")
