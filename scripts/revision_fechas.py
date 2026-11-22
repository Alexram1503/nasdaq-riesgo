import pandas as pd
from pathlib import Path

# ============================================================
# 1. RUTA DE LA BASE TRANSFORMADA
# ============================================================

RUTA_BASE = (
    Path(__file__).resolve().parent.parent  # raiz del repositorio
    / "data" / "processed" / "base_transformada.csv"
)

if not RUTA_BASE.exists():
    raise FileNotFoundError(f"No se encontró el archivo: {RUTA_BASE}")

# ============================================================
# 2. CARGA DE DATOS
# ============================================================

datos = pd.read_csv(RUTA_BASE, sep=None, engine="python")

print("\n" + "=" * 65)
print("REVISIÓN DE LA ESTRUCTURA TEMPORAL")
print("=" * 65)

print("\nDimensiones de la base:")
print(f"Filas: {datos.shape[0]}")
print(f"Columnas: {datos.shape[1]}")

print("\nColumnas disponibles:")
print(datos.columns.tolist())

# ============================================================
# 3. IDENTIFICAR LA COLUMNA DE FECHA
# ============================================================

columna_fecha = next(
    (col for col in datos.columns if col.strip().lower() in
     ["fecha", "date", "datetime", "time"]),
    None
)

if columna_fecha is None:
    raise ValueError(
        "No se encontró una columna de fecha. "
        "Revisa los nombres de las columnas."
    )

fechas_originales = datos[columna_fecha].astype(str).str.strip()

print(f"\nColumna de fecha identificada: {columna_fecha}")

print("\nPrimeros 10 valores originales:")
print(fechas_originales.head(10).to_string(index=False))

print("\nÚltimos 10 valores originales:")
print(fechas_originales.tail(10).to_string(index=False))

# ============================================================
# 4. COMPARAR LAS DOS INTERPRETACIONES DE FECHA
# ============================================================

fecha_dayfirst = pd.to_datetime(
    fechas_originales,
    errors="coerce",
    dayfirst=True,
    format="mixed"
)

fecha_monthfirst = pd.to_datetime(
    fechas_originales,
    errors="coerce",
    dayfirst=False,
    format="mixed"
)

def evaluar_fechas(fechas, nombre):
    validas = fechas.dropna()

    if validas.empty:
        return {
            "Interpretación": nombre,
            "Fechas válidas": 0,
            "Primera fecha": None,
            "Última fecha": None,
            "Meses distintos": 0,
            "Meses duplicados": 0,
            "Meses faltantes": "No evaluable"
        }

    meses = validas.dt.to_period("M")
    meses_unicos = meses.drop_duplicates().sort_values()

    rango_meses = pd.period_range(
        start=meses_unicos.min(),
        end=meses_unicos.max(),
        freq="M"
    )

    faltantes = rango_meses.difference(meses_unicos)

    return {
        "Interpretación": nombre,
        "Fechas válidas": len(validas),
        "Primera fecha": validas.min(),
        "Última fecha": validas.max(),
        "Meses distintos": meses.nunique(),
        "Meses duplicados": int(meses.duplicated().sum()),
        "Meses faltantes": len(faltantes)
    }

resultados = pd.DataFrame([
    evaluar_fechas(fecha_dayfirst, "Día/Mes/Año"),
    evaluar_fechas(fecha_monthfirst, "Mes/Día/Año")
])

print("\n" + "=" * 65)
print("COMPARACIÓN DE INTERPRETACIONES")
print("=" * 65)
print(resultados.to_string(index=False))

# ============================================================
# 5. SELECCIONAR LA INTERPRETACIÓN PARA LA REVISIÓN
# ============================================================

# Si las fechas están escritas en formato ISO (AAAA-MM-DD),
# ambas interpretaciones producen el mismo resultado.
# Si son ambiguas, mostramos ambas y usamos día/mes/año
# provisionalmente para el diagnóstico.

if fecha_dayfirst.equals(fecha_monthfirst):
    fechas = fecha_dayfirst
    interpretacion = "Ambas interpretaciones coinciden"
else:
    fechas = fecha_dayfirst
    interpretacion = "Día/Mes/Año (provisional; revisar comparación)"

print("\nInterpretación utilizada:")
print(interpretacion)

if fechas.isna().any():
    print(
        f"\nADVERTENCIA: hay {fechas.isna().sum()} "
        "fechas que no pudieron convertirse."
    )

# ============================================================
# 6. ORDEN CRONOLÓGICO Y FECHAS REPETIDAS
# ============================================================

print("\n" + "=" * 65)
print("ORDEN Y DUPLICADOS")
print("=" * 65)

fechas_validas = fechas.dropna()

print(
    "¿Las fechas están ordenadas cronológicamente?:",
    fechas_validas.is_monotonic_increasing
)

print(
    "¿Hay fechas exactas duplicadas?:",
    fechas_validas.duplicated().any()
)

print(
    "Cantidad de fechas exactas duplicadas:",
    fechas_validas.duplicated().sum()
)

# ============================================================
# 7. REVISIÓN DE LA FRECUENCIA MENSUAL
# ============================================================

print("\n" + "=" * 65)
print("FRECUENCIA MENSUAL")
print("=" * 65)

meses = fechas_validas.dt.to_period("M")

conteo_meses = meses.value_counts().sort_index()

print("\nObservaciones por mes:")
print(conteo_meses.value_counts().sort_index().to_string())

print(
    "\n¿Existe exactamente una observación por cada mes?:",
    bool((conteo_meses == 1).all())
)

if not meses.empty:
    rango_completo = pd.period_range(
        start=meses.min(),
        end=meses.max(),
        freq="M"
    )

    meses_faltantes = rango_completo.difference(meses.unique())
    meses_duplicados = conteo_meses[conteo_meses > 1]

    print("\nMeses faltantes:")
    print(
        meses_faltantes.astype(str).tolist()
        if len(meses_faltantes) > 0
        else "Ninguno"
    )

    print("\nMeses con más de una observación:")
    print(
        meses_duplicados.to_string()
        if not meses_duplicados.empty
        else "Ninguno"
    )

# ============================================================
# 8. MOSTRAR LAS FECHAS YA INTERPRETADAS
# ============================================================

print("\n" + "=" * 65)
print("FECHAS INTERPRETADAS")
print("=" * 65)

revision = pd.DataFrame({
    "Fecha original": fechas_originales,
    "Fecha interpretada": fechas
})

print("\nPrimeras 10 observaciones:")
print(revision.head(10).to_string(index=False))

print("\nÚltimas 10 observaciones:")
print(revision.tail(10).to_string(index=False))

# ============================================================
# 9. DIFERENCIAS ENTRE MESES
# ============================================================

print("\n" + "=" * 65)
print("CONTINUIDAD TEMPORAL")
print("=" * 65)

meses_ordenados = pd.PeriodIndex(
    meses.unique()
).sort_values()

if len(meses_ordenados) > 1:
    diferencias = [
        (meses_ordenados[i].year - meses_ordenados[i - 1].year) * 12
        + meses_ordenados[i].month - meses_ordenados[i - 1].month
        for i in range(1, len(meses_ordenados))
    ]

    print("Saltos entre observaciones mensuales:")
    print(pd.Series(diferencias).value_counts().sort_index().to_string())

    print(
        "\n¿Todos los saltos entre meses son de un mes?:",
        all(d == 1 for d in diferencias)
    )

# ============================================================
# 10. CONCLUSIÓN AUTOMÁTICA
# ============================================================

print("\n" + "=" * 65)
print("CONCLUSIÓN PRELIMINAR")
print("=" * 65)

if (
    fechas.notna().all()
    and fechas_validas.is_monotonic_increasing
    and not fechas_validas.duplicated().any()
    and len(meses_faltantes) == 0
    and len(meses_duplicados) == 0
):
    print("La estructura temporal parece consistente:")
    print("- Todas las fechas pudieron interpretarse.")
    print("- Las fechas están ordenadas.")
    print("- No hay fechas repetidas.")
    print("- Existe una observación por mes.")
    print("- No hay meses faltantes.")
else:
    print("Se encontraron aspectos que requieren revisión.")
    print("No debemos dar por validada todavía la estructura temporal.")

print("\nREVISIÓN FINALIZADA.")