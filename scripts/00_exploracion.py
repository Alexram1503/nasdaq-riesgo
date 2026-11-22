
# ==========================================================
# PROYECTO INTEGRADOR: DINAMICA DEL NASDAQ
# ARCHIVO 00: EXPLORACION DE DATOS
# Autor: Equipo Proyecto Integrador
# Objetivo: Explorar y validar la base de datos original
# ==========================================================


# 1. IMPORTAR LIBRERIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# 2. DEFINIR RUTAS

# Ruta principal del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent  # carpeta raiz del repositorio (ruta relativa)

# Ruta del archivo CSV
RUTA_DATOS = BASE_DIR / "data" / "raw" / "final.csv"

# Rutas de almacenamiento
RUTA_TABLAS = BASE_DIR / "outputs" / "tablas"
RUTA_GRAFICOS = BASE_DIR / "outputs" / "graficos"
RUTA_PROCESADOS = BASE_DIR / "data" / "processed"

# Crear carpetas automaticamente
RUTA_TABLAS.mkdir(parents=True, exist_ok=True)
RUTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)


# 3. CARGAR BASE DE DATOS

print("\nCARGANDO BASE DE DATOS...")

df = pd.read_csv(
    RUTA_DATOS,
    sep=";",
    decimal=","
)

# Limpiar nombres de columnas
df.columns = df.columns.str.strip()

print("\nBASE CARGADA CORRECTAMENTE")


# 4. LIMPIEZA INICIAL

# Eliminar columnas de indices adicionales
columnas_indice = [
    col for col in df.columns
    if col.lower().startswith("unnamed")
]

df = df.drop(columns=columnas_indice)

# Convertir fecha
df["Fecha"] = pd.to_datetime(
    df["Fecha"],
    dayfirst=True,
    errors="coerce"
)

# Variables del proyecto
variables = [
    "NASDAQCOM",
    "WTI",
    "BAA10YM",
    "GS10"
]

# Convertir variables a formato numerico
for variable in variables:

    df[variable] = pd.to_numeric(
        df[variable],
        errors="coerce"
    )


# 5. INFORMACION GENERAL

print("\n======================================")
print("5. INFORMACION GENERAL")
print("======================================")

df.info()

print("\nPrimeras observaciones:")
print(df.head())

print("\nUltimas observaciones:")
print(df.tail())


# 6. DIMENSIONES DE LA BASE

print("\n======================================")
print("6. DIMENSIONES")
print("======================================")

print("Numero de filas:", df.shape[0])
print("Numero de columnas:", df.shape[1])


# 7. VALORES FALTANTES

print("\n======================================")
print("7. VALORES FALTANTES")
print("======================================")

faltantes = df.isnull().sum()

print(faltantes)

faltantes.to_csv(
    RUTA_TABLAS / "valores_faltantes.csv"
)


# 8. REGISTROS DUPLICADOS

print("\n======================================")
print("8. REGISTROS DUPLICADOS")
print("======================================")

print(
    "Filas duplicadas:",
    df.duplicated().sum()
)

print(
    "Fechas duplicadas:",
    df["Fecha"].duplicated().sum()
)


# 9. PERIODO TEMPORAL

print("\n======================================")
print("9. PERIODO DE ANALISIS")
print("======================================")

print("Fecha inicial:", df["Fecha"].min())

print("Fecha final:", df["Fecha"].max())

print("Numero de observaciones:", len(df))

print("\nFrecuencia entre fechas:")

print(
    df["Fecha"]
    .sort_values()
    .diff()
    .value_counts()
    .head()
)


# 10. ESTADISTICAS DESCRIPTIVAS

print("\n======================================")
print("10. ESTADISTICAS DESCRIPTIVAS")
print("======================================")

estadisticas = df[variables].describe().T

print(estadisticas)

estadisticas.to_csv(
    RUTA_TABLAS / "estadisticas_descriptivas.csv"
)


# 11. GRAFICOS HISTORICOS

print("\n======================================")
print("11. GRAFICOS HISTORICOS")
print("======================================")

for variable in variables:

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Fecha"],
        df[variable],
        linewidth=1.2
    )

    plt.title(
        f"Evolucion historica de {variable}"
    )

    plt.xlabel("Fecha")
    plt.ylabel(variable)

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RUTA_GRAFICOS / f"{variable}_historico.png",
        dpi=300
    )

    plt.show()

    plt.close()


# 12. MATRIZ DE CORRELACIONES

print("\n======================================")
print("12. MATRIZ DE CORRELACIONES")
print("======================================")

correlaciones = df[variables].corr()

print(correlaciones)

correlaciones.to_csv(
    RUTA_TABLAS / "matriz_correlaciones.csv"
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlaciones,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title(
    "Matriz de correlacion de las variables"
)

plt.tight_layout()

plt.savefig(
    RUTA_GRAFICOS / "matriz_correlaciones.png",
    dpi=300
)

plt.show()

plt.close()


# 13. ORDENAR Y GUARDAR BASE EXPLORATORIA

df = df.sort_values("Fecha").reset_index(drop=True)

df.to_csv(
    RUTA_PROCESADOS / "base_exploracion.csv",
    index=False
)


# 14. RESUMEN FINAL

print("\n======================================")
print("EXPLORACION FINALIZADA")
print("======================================")

print("Base original:", RUTA_DATOS)

print("Base exploratoria guardada.")

print("Tablas guardadas en:", RUTA_TABLAS)

print("Graficos guardados en:", RUTA_GRAFICOS)

print("\nObservaciones:", len(df))

print("Variables analizadas:", variables)

print("\nPROCESO COMPLETADO")
