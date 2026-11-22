
# ============================================================
# PROYECTO INTEGRADOR: DINAMICA DEL NASDAQ
# ARCHIVO 01: TRANSFORMACIONES Y ESTACIONARIEDAD
#
# Objetivo:
# Transformar las series originales y evaluar su
# estacionariedad antes de estimar el modelo VAR.
# ============================================================


# 1. IMPORTAR LIBRERIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from statsmodels.tsa.stattools import adfuller, kpss


# 2. DEFINIR RUTAS

BASE_DIR = Path(__file__).resolve().parent.parent  # carpeta raiz del repositorio (ruta relativa)

RUTA_ENTRADA = (
    BASE_DIR / "data" / "processed" /
    "base_exploracion.csv"
)

RUTA_TABLAS = BASE_DIR / "outputs" / "tablas"

RUTA_GRAFICOS = BASE_DIR / "outputs" / "graficos"

RUTA_PROCESADOS = BASE_DIR / "data" / "processed"

# Crear carpetas si no existen

RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)


# 3. CARGAR BASE EXPLORATORIA

print("\n======================================")
print("CARGANDO BASE DE DATOS")
print("======================================")

df = pd.read_csv(
    RUTA_ENTRADA,
    parse_dates=["Fecha"]
)

df = df.sort_values("Fecha").reset_index(drop=True)

print(df.head())

print("\nObservaciones originales:", len(df))


# 4. VERIFICAR VARIABLES

variables = [
    "NASDAQCOM",
    "WTI",
    "BAA10YM",
    "GS10"
]

for variable in variables:

    if variable not in df.columns:
        raise ValueError(
            f"No se encuentra la variable {variable}"
        )

    if df[variable].isna().any():
        raise ValueError(
            f"La variable {variable} contiene valores faltantes"
        )

print("\nVariables verificadas correctamente.")


# 5. TRANSFORMACIONES

print("\n======================================")
print("CALCULANDO TRANSFORMACIONES")
print("======================================")

df_transformado = pd.DataFrame()

df_transformado["Fecha"] = df["Fecha"]

# 5.1 Nasdaq: rentabilidad logaritmica

if (df["NASDAQCOM"] <= 0).any():
    raise ValueError("NASDAQCOM debe ser positivo para usar logaritmos.")

df_transformado["NASDAQ_RET"] = (
    100 * np.log(df["NASDAQCOM"] / df["NASDAQCOM"].shift(1))
)

# 5.2 WTI: rentabilidad logaritmica

if (df["WTI"] <= 0).any():
    raise ValueError("WTI debe ser positivo para usar logaritmos.")

df_transformado["WTI_RET"] = (
    100 * np.log(df["WTI"] / df["WTI"].shift(1))
)

# 5.3 BAA10YM: conservar en niveles
# Las pruebas ADF y KPSS indican estacionariedad.

df_transformado["BAA10YM"] = df["BAA10YM"]

# 5.4 GS10: primera diferencia

df_transformado["GS10_DIF"] = (
    df["GS10"].diff()
)

# 6. ELIMINAR PRIMERA OBSERVACION SIN TRANSFORMACION

# Las diferencias y rentabilidades necesitan
# la observacion del mes anterior.

df_transformado = df_transformado.dropna().reset_index(drop=True)

print("\nTransformaciones calculadas.")

print("Observaciones originales:", len(df))

print("Observaciones transformadas:", len(df_transformado))

print("\nPrimeras observaciones transformadas:")

print(df_transformado.head())


# 7. GRAFICOS INDIVIDUALES

variables_transformadas = [
    "NASDAQ_RET",
    "WTI_RET",
    "BAA10YM",
    "GS10_DIF"
]

nombres = {
    "NASDAQ_RET": "Rentabilidad del Nasdaq",
    "WTI_RET": "Rentabilidad del WTI",
    "BAA10YM": "Diferencial BAA en niveles",
    "GS10_DIF": "Primera diferencia de GS10"
}

print("\n======================================")
print("GENERANDO GRAFICOS")
print("======================================")

for variable in variables_transformadas:

    plt.figure(figsize=(12, 5))

    plt.plot(
        df_transformado["Fecha"],
        df_transformado[variable],
        linewidth=1
    )

    plt.axhline(
        y=0,
        linestyle="--",
        alpha=0.6
    )

    plt.title(nombres[variable])

    plt.xlabel("Fecha")

    plt.ylabel("Variacion")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RUTA_GRAFICOS / f"{variable}_transformada.png",
        dpi=300
    )

    plt.show()

    plt.close()


# 8. GRAFICO CONJUNTO

fig, axes = plt.subplots(
    2, 2,
    figsize=(15, 9)
)

for ax, variable in zip(
    axes.flatten(),
    variables_transformadas
):

    ax.plot(
        df_transformado["Fecha"],
        df_transformado[variable],
        linewidth=0.9
    )

    ax.axhline(
        y=0,
        linestyle="--",
        alpha=0.5
    )

    ax.set_title(nombres[variable])

    ax.grid(alpha=0.3)

plt.suptitle(
    "Series de tiempo transformadas",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    RUTA_GRAFICOS / "series_transformadas_conjunto.png",
    dpi=300
)

plt.show()

plt.close()


# 9. FUNCION PARA PRUEBAS DE ESTACIONARIEDAD

def pruebas_estacionariedad(serie, nombre):

    # Eliminar valores faltantes
    serie = serie.dropna()

    # Prueba ADF
    resultado_adf = adfuller(
        serie,
        autolag="AIC",
        regression="c"
    )

    estadistico_adf = resultado_adf[0]
    pvalor_adf = resultado_adf[1]
    rezagos_adf = resultado_adf[2]

    # Prueba KPSS
    resultado_kpss = kpss(
        serie,
        regression="c",
        nlags="auto"
    )

    estadistico_kpss = resultado_kpss[0]
    pvalor_kpss = resultado_kpss[1]
    rezagos_kpss = resultado_kpss[2]

    # Interpretacion individual
    if pvalor_adf < 0.05:
        decision_adf = "Rechaza raiz unitaria"
    else:
        decision_adf = "No rechaza raiz unitaria"

    if pvalor_kpss < 0.05:
        decision_kpss = "Rechaza estacionariedad"
    else:
        decision_kpss = "No rechaza estacionariedad"

    return {
        "Variable": nombre,
        "ADF_estadistico": estadistico_adf,
        "ADF_pvalor": pvalor_adf,
        "ADF_rezagos": rezagos_adf,
        "ADF_decision": decision_adf,
        "KPSS_estadistico": estadistico_kpss,
        "KPSS_pvalor": pvalor_kpss,
        "KPSS_rezagos": rezagos_kpss,
        "KPSS_decision": decision_kpss
    }


# 10. EVALUAR SERIES ORIGINALES Y TRANSFORMADAS

print("\n======================================")
print("PRUEBAS DE ESTACIONARIEDAD")
print("======================================")

resultados = []

# Primero evaluamos niveles originales

for variable in variables:

    resultado = pruebas_estacionariedad(
        df[variable],
        variable + "_NIVEL"
    )

    resultados.append(resultado)

# Despues evaluamos las transformaciones

for variable in variables_transformadas:

    resultado = pruebas_estacionariedad(
        df_transformado[variable],
        variable
    )

    resultados.append(resultado)


# 11. CONSOLIDAR RESULTADOS

tabla_estacionariedad = pd.DataFrame(resultados)

print("\nRESULTADOS ADF Y KPSS")

print(
    tabla_estacionariedad.to_string(index=False)
)

tabla_estacionariedad.to_csv(
    RUTA_TABLAS / "resultados_estacionariedad.csv",
    index=False
)


# 12. GUARDAR BASE TRANSFORMADA

df_transformado.to_csv(
    RUTA_PROCESADOS / "base_transformada.csv",
    index=False
)


# 13. RESUMEN FINAL

print("\n======================================")
print("PROCESO DE TRANSFORMACION FINALIZADO")
print("======================================")

print("Observaciones originales:", len(df))

print("Observaciones transformadas:", len(df_transformado))

print(
    "Base guardada en:",
    RUTA_PROCESADOS / "base_transformada.csv"
)

print(
    "Resultados de estacionariedad:",
    RUTA_TABLAS / "resultados_estacionariedad.csv"
)

print(
    "Graficos guardados en:",
    RUTA_GRAFICOS
)

print("\nPROCESO COMPLETADO")
