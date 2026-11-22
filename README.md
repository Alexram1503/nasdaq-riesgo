# Monitor de riesgo del Nasdaq (1995-2026)

Proyecto integrador del curso **Análisis e Interpretación de Datos a través de Python**.

Analiza la dinámica conjunta del Nasdaq Composite, la tasa del Tesoro a 10 años, el petróleo WTI y el diferencial de crédito Baa, y estima la probabilidad de que el Nasdaq entre en un estado de **Estrés** en los próximos 12 meses. El resultado apoya una regla de alerta para un equipo de Middle Office.

**Aplicación publicada:** _(pegar aquí el link de Streamlit)_

## Métodos

| Módulo | Qué hace |
|---|---|
| VAR(2) | Dinámica conjunta de las 4 variables, causalidad de Granger, impulso-respuesta y pronóstico a 12 meses |
| Markov | Tres estados del Nasdaq según el drawdown frente al máximo de 12 meses: Normal (< 5%), Vulnerable (5-10%), Estrés (≥ 10%) |
| Monte Carlo | 5.000 trayectorias de 12 meses generadas con el VAR(2) y residuos remuestreados (bootstrap); cada mes se clasifica con los estados de Markov |

## Datos

Series mensuales de FRED (Federal Reserve Bank of St. Louis), enero 1995 a septiembre 2026:

| Serie | Variable | Unidad |
|---|---|---|
| NASDAQCOM | Nasdaq Composite | Índice |
| GS10 | Rendimiento del Tesoro a 10 años | % |
| MCOILWTICO | Petróleo WTI | USD por barril |
| BAA10YM | Diferencial Baa - Tesoro 10 años | Puntos porcentuales |

La base original está en `data/raw/final.csv` (separador `;`, decimal `,`).

## Estructura

```
├── app.py                    Aplicación Streamlit (lee los resultados de outputs/)
├── requirements.txt          Librerías necesarias
├── data/
│   ├── raw/final.csv         Base original
│   └── processed/            Bases generadas por los scripts 00 y 01
├── scripts/
│   ├── 00_exploracion.py     Limpieza, faltantes, descriptivos
│   ├── 01_transformaciones.py  Rentabilidades, diferencias, ADF y KPSS
│   ├── 02_var.py             Rezagos, VAR(2), diagnósticos, Granger, IRF, FEVD, pronóstico
│   ├── 03_markov.py          Estados, matriz de transición, proyección
│   ├── 04_montecarlo.py      Simulación basada en el VAR(2)
│   └── revision_fechas.py    Verificación de la columna de fechas
└── outputs/
    ├── tablas/               Resultados en CSV
    └── graficos/             Gráficos en PNG
```

## Cómo ejecutarlo

```bash
# 1. Instalar dependencias
python -m pip install -r requirements.txt

# 2. (Opcional) Regenerar todos los resultados, en orden
python scripts/00_exploracion.py
python scripts/01_transformaciones.py
python scripts/02_var.py
python scripts/03_markov.py
python scripts/04_montecarlo.py

# 3. Abrir la aplicación
python -m streamlit run app.py
```

Todas las rutas son relativas a la carpeta del proyecto, así que funciona en cualquier computador. La simulación usa la semilla `12345`, por lo que los resultados son reproducibles.

## Publicación

1. Subir este repositorio a GitHub (público).
2. En [share.streamlit.io](https://share.streamlit.io) → **Create app** → elegir el repositorio, la rama `main` y el archivo `app.py`.

## Límites

- Los resultados muestran asociación temporal y causalidad de Granger, no causalidad estructural.
- El VAR supone volatilidad constante; la prueba ARCH lo rechaza para el Nasdaq y el WTI.
- La cadena de Markov es de primer orden y su matriz no es idéntica entre submuestras.
- El WTI es un proxy energético general.

## Integrantes

- Wilmer Fabián Prieto Rojas: datos y ETL
- _(Integrante 2)_: modelo VAR
- _(Integrante 3)_: Markov, Monte Carlo, Streamlit e interpretación
