# ============================================================
# MONITOR DE RIESGO DEL NASDAQ  -  Aplicacion Streamlit
# Proyecto Integrador: Analisis e Interpretacion de Datos con Python
#
# La app NO re-estima los modelos: lee las tablas y graficos que
# generan los scripts 00-04 (carpeta scripts/). Asi carga rapido en
# Streamlit Cloud y muestra exactamente los resultados del equipo.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).resolve().parent
TABLAS = BASE / "outputs" / "tablas"
GRAFICOS = BASE / "outputs" / "graficos"
PROCESADOS = BASE / "data" / "processed"

ESTADOS = ["Normal", "Vulnerable", "Estres"]
COLORES = {"Normal": "#2e7d32", "Vulnerable": "#f9a825", "Estres": "#c62828"}
NOMBRES = {
    "NASDAQCOM": "Nasdaq Composite (indice)",
    "GS10": "Bono del Tesoro 10 años (%)",
    "WTI": "Petróleo WTI (USD/barril)",
    "BAA10YM": "Diferencial Baa - Tesoro 10a (p.p.)",
}

st.set_page_config(page_title="Monitor de riesgo del Nasdaq", layout="wide")


@st.cache_data
def tabla(nombre, **kwargs):
    return pd.read_csv(TABLAS / nombre, encoding="utf-8-sig", **kwargs)


@st.cache_data
def base(nombre):
    return pd.read_csv(PROCESADOS / nombre, parse_dates=["Fecha"])


def imagen(nombre, pie=None):
    ruta = GRAFICOS / nombre
    if ruta.exists():
        st.image(str(ruta), caption=pie, width="stretch")
    else:
        st.warning(f"No se encontró el gráfico {nombre}. Ejecuta los scripts 00-04.")


def pct(x, dec=1):
    """Porcentaje con coma decimal (formato colombiano)."""
    return f"{100 * x:.{dec}f}%".replace(".", ",")


# ---------- datos que se usan en varias secciones ----------
niveles = base("base_exploracion.csv")
transformada = base("base_transformada.csv")
secuencia = tabla("markov_secuencia_estados.csv", parse_dates=["Fecha"])
matriz = tabla("markov_matriz_transicion.csv", index_col=0)
proy_markov = tabla("markov_proyeccion_12m.csv")
mc = tabla("mc_resumen_horizontes.csv", parse_dates=["Fecha"])
mc_met = tabla("mc_metricas.csv").set_index("Indicador")["Valor"]

estado_actual = secuencia["Nombre_estado"].iloc[-1]
fecha_actual = secuencia["Fecha"].iloc[-1]

# ---------- barra lateral ----------
st.sidebar.title("Monitor de riesgo del Nasdaq")
seccion = st.sidebar.radio(
    "Sección",
    ["Inicio", "Datos", "VAR", "Markov", "Monte Carlo", "Conclusiones"],
)
st.sidebar.markdown("---")
st.sidebar.subheader("Regla de alerta")
st.sidebar.caption("Alerta si la probabilidad de estar en Estrés en el mes h, "
                   "según Monte Carlo, supera el umbral. Lo habitual es 22,6%.")
metrica_alerta = "Estar en Estrés en el mes h"
horizonte = st.sidebar.slider("Horizonte h (meses)", 1, 12, 12)
umbral = st.sidebar.slider("Umbral de alerta", 0.10, 0.90, 0.30, 0.05)

col_mc = "P_Estres"
p_mc = mc.loc[mc["Mes_horizonte"] == horizonte, col_mc].iloc[0]
p_mk = proy_markov.loc[proy_markov["Mes_horizonte"] == horizonte, col_mc].iloc[0]
alerta = p_mc > umbral

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Datos mensuales FRED 1995-01 a {fecha_actual:%Y-%m}. "
    "Equipo Proyecto Integrador."
)


def tarjeta_alerta():
    if alerta:
        st.error(
            f"🔴 **ALERTA ACTIVA** · P(Monte Carlo) = {pct(p_mc)} supera el umbral de "
            f"{pct(umbral, 0)} ({metrica_alerta.lower()}, h = {horizonte})."
        )
    else:
        st.success(
            f"🟢 **Sin alerta** · P(Monte Carlo) = {pct(p_mc)} está por debajo del umbral de "
            f"{pct(umbral, 0)} ({metrica_alerta.lower()}, h = {horizonte})."
        )


# ============================================================
# INICIO
# ============================================================
if seccion == "Inicio":
    st.title("Dinámica del Nasdaq y riesgo financiero (1995-2026)")
    st.markdown(
        """
**Usuario:** equipo de Middle Office de una firma que invierte en el mercado estadounidense.

**Decisión:** emitir una alerta de riesgo al Front Office y al Comité de Riesgos cuando
la probabilidad de que el Nasdaq entre en estado de **Estrés** (caída de 10% o más frente a
su máximo de los últimos 12 meses) supere un umbral, para ajustar la exposición tecnológica
o activar coberturas.

**Pregunta central:** ¿cómo se relacionan dinámicamente el Nasdaq Composite, la tasa del
Tesoro a 10 años, el petróleo WTI y el diferencial de crédito Baa, y qué probabilidad hay de
que el Nasdaq pase a estados de vulnerabilidad o estrés en los próximos 12 meses?
"""
    )
    tarjeta_alerta()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Estado actual ({fecha_actual:%m/%Y})", estado_actual)
    c2.metric("Drawdown 12m actual", pct(secuencia["Drawdown_12m"].iloc[-1]))
    c3.metric(f"P Monte Carlo (h={horizonte})", pct(p_mc))
    c4.metric(f"P Markov (h={horizonte})", pct(p_mk))

    st.subheader("Cómo se conectan los tres métodos")
    st.markdown(
        """
1. **VAR(2)** estima cómo se mueven juntas las cuatro variables mes a mes y genera el pronóstico base.
2. **Markov** convierte el Nasdaq en tres estados de riesgo y mide qué tan persistentes son.
3. **Monte Carlo** usa el VAR para simular 5.000 futuros posibles, clasifica cada uno con
   los estados de Markov y calcula la probabilidad de Estrés que activa o no la alerta.
"""
    )
    st.info(
        "La regla de alerta se puede ajustar en la barra lateral. Los resultados describen "
        "dinámica y escenarios; no demuestran causalidad estructural ni predicen crisis."
    )

# ============================================================
# DATOS
# ============================================================
elif seccion == "Datos":
    st.title("Datos")
    st.markdown(
        f"""
Cuatro series mensuales de **FRED** (Federal Reserve Bank of St. Louis),
{niveles['Fecha'].min():%Y-%m} a {niveles['Fecha'].max():%Y-%m}: **{len(niveles)} meses**, sin valores faltantes.
"""
    )
    st.dataframe(
        pd.DataFrame({
            "Serie FRED": ["NASDAQCOM", "GS10", "MCOILWTICO", "BAA10YM"],
            "Variable": list(NOMBRES.values()),
            "Rol": ["Objetivo (estados de riesgo)", "Condiciones monetarias",
                    "Costos energéticos (proxy)", "Riesgo de crédito"],
            "Transformación para el VAR": [
                "Rentabilidad log ×100", "Primera diferencia",
                "Rentabilidad log ×100", "Niveles (estacionaria)"],
        }),
        hide_index=True, width="stretch",
    )

    variable = st.selectbox("Serie a graficar", list(NOMBRES), format_func=NOMBRES.get)
    log = st.checkbox("Escala logarítmica", value=(variable == "NASDAQCOM"))
    fig = px.line(niveles, x="Fecha", y=variable, log_y=log,
                  labels={variable: NOMBRES[variable]})
    if log:
        fig.update_yaxes(dtick=1, tickformat=",d")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Estacionariedad (ADF y KPSS)")
    st.caption("ADF: H0 = raíz unitaria. KPSS: H0 = estacionariedad. "
               "Se busca que ADF rechace y KPSS no rechace.")
    est = tabla("resultados_estacionariedad.csv")
    st.dataframe(
        est[["Variable", "ADF_pvalor", "ADF_decision", "KPSS_pvalor", "KPSS_decision"]]
        .round(4), hide_index=True, width="stretch",
    )
    st.markdown(
        "Nasdaq, WTI y GS10 no son estacionarias en niveles, por eso entran al VAR como "
        "rentabilidades o diferencias. El diferencial Baa sí lo es en niveles y se conserva así. "
        "La transformación cuesta una observación: el VAR usa 380 meses."
    )
    imagen("matriz_correlaciones.png", "Correlaciones en niveles (solo descriptivas)")

# ============================================================
# VAR
# ============================================================
elif seccion == "VAR":
    st.title("Modelo VAR(2)")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Selección de rezagos")
        st.dataframe(tabla("seleccion_rezagos.csv"), hide_index=True)
        st.caption("AIC, HQIC y FPE eligen 2 rezagos; BIC elige 1. Se usa VAR(2) y se "
                   "comparó con VAR(1) en la misma muestra.")
    with c2:
        st.subheader("Diagnósticos")
        raices = tabla("raices_VAR2.csv")
        arch = tabla("pruebas_ARCH_VAR2.csv")
        raiz_min = f"{raices.iloc[:, 1].min():.3f}".replace(".", ",")
        st.markdown(
            f"""
- **Estabilidad:** módulo mínimo de las raíces = {raiz_min} (> 1 → estable
  en la convención de statsmodels).
- **Autocorrelación de residuos** (Portmanteau, 12 rezagos): p = 0,078 → no se rechaza ruido blanco.
- **Normalidad:** p < 0,001 → se rechaza (colas pesadas). Por eso Monte Carlo usa *bootstrap*.
- **ARCH:** se rechaza homocedasticidad en Nasdaq y WTI
  (p < 0,0001 en ambos). La volatilidad cambia en el tiempo.
"""
        )

    st.subheader("Causalidad de Granger")
    gr = tabla("causalidad_Granger_VAR2.csv")
    gr["p-valor"] = gr["p-valor"].map(lambda p: "< 0,0001" if p < 1e-4 else f"{p:.4f}".replace(".", ","))
    st.dataframe(gr.round(3), hide_index=True, width="stretch")
    st.warning(
        "Granger es causalidad **predictiva** dentro del modelo: los rezagos de una variable "
        "ayudan a pronosticar otra. No prueba causalidad estructural. Aquí el Nasdaq anticipa al "
        "WTI, al diferencial Baa y a GS10, pero ninguna de las tres anticipa al Nasdaq."
    )

    st.subheader("Pronóstico a 12 meses (intervalo 95%)")
    pr = tabla("pronosticos_VAR2_intervalos_95.csv", index_col=0, parse_dates=True)
    var = st.selectbox("Variable", ["NASDAQ_RET", "WTI_RET", "BAA10YM", "GS10_DIF"])
    hist = transformada.set_index("Fecha")[var].iloc[-48:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observado", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=pr.index, y=pr[f"{var}_Superior_95"], mode="lines", line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=pr.index, y=pr[f"{var}_Inferior_95"], fill="tonexty", mode="lines",
                             fillcolor="rgba(21,101,192,0.2)", line=dict(width=0),
                             name="IC 95%"))
    fig.add_trace(go.Scatter(x=pr.index, y=pr[f"{var}_Pronostico"], name="Pronóstico", mode="lines",
                             line=dict(color="#1565c0")))
    st.plotly_chart(fig, width="stretch")
    st.caption("El pronóstico del Nasdaq converge rápido a su media (~0,9% mensual) y el "
               "intervalo es muy amplio: el VAR aporta poco al nivel esperado, pero sí "
               "aporta la estructura de correlación que usa Monte Carlo.")

    with st.expander("Impulso-respuesta y descomposición de varianza"):
        imagen("funciones_IRF_VAR2.png", "Funciones impulso-respuesta (forma reducida)")
        imagen("descomposicion_FEVD_VAR2.png",
               "FEVD: depende del orden de Cholesky NASDAQ_RET → WTI_RET → BAA10YM → GS10_DIF")
    with st.expander("Residuos"):
        imagen("residuos_temporales_VAR2.png")
        imagen("qqplots_residuos_VAR2.png")

# ============================================================
# MARKOV
# ============================================================
elif seccion == "Markov":
    st.title("Cadenas de Markov: estados de riesgo")
    st.markdown(
        """
**Indicador:** drawdown frente al máximo de los últimos 12 meses,
$DD_t = 1 - P_t / \\max(P_{t-11},\\dots,P_t)$.

| Estado | Criterio |
|---|---|
| Normal | DD < 5% |
| Vulnerable | 5% ≤ DD < 10% |
| Estrés | DD ≥ 10% |

Se usa el máximo de 12 meses y no el máximo histórico porque con este último el Nasdaq
habría estado en Estrés casi sin interrupción entre 2000 y 2015, cuando aún no recuperaba
el pico de la burbuja puntocom.
"""
    )
    fig = px.scatter(secuencia, x="Fecha", y="NASDAQCOM", color="Nombre_estado",
                     color_discrete_map=COLORES, log_y=True,
                     category_orders={"Nombre_estado": ESTADOS},
                     labels={"NASDAQCOM": "Nasdaq (log)", "Nombre_estado": "Estado"})
    fig.update_traces(marker=dict(size=5))
    fig.update_yaxes(dtick=1, tickformat=",d")
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Matriz de transición mensual")
        st.caption("Filas = estado actual · columnas = estado del mes siguiente")
        fig = px.imshow(matriz, text_auto=".2f", color_continuous_scale="Blues",
                        zmin=0, zmax=1, labels=dict(x="Mes siguiente", y="Estado actual"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Persistencia")
        ind = tabla("markov_indicadores.csv", index_col=0)
        st.dataframe(ind.round(3), width="stretch")
        est = tabla("markov_distribucion_estacionaria.csv", index_col=0)
        st.dataframe(est.round(3), width="stretch")
        st.caption("La distribución estacionaria coincide con la frecuencia observada.")

    st.markdown(
        f"""
**Lectura:** Normal y Estrés son estados persistentes (permanencia de
{pct(matriz.loc['Normal', 'Normal'])} y {pct(matriz.loc['Estres', 'Estres'])}).
Vulnerable es un estado de paso: dura en promedio
{ind.loc['Vulnerable', 'Duracion_esperada_meses']:.1f} meses y desde ahí el mercado
se reparte entre volver a Normal ({pct(matriz.loc['Vulnerable', 'Normal'])}) o pasar a Estrés
({pct(matriz.loc['Vulnerable', 'Estres'])}). Esto describe frecuencias, no mecanismos causales.
"""
    )

    st.subheader(f"Proyección desde el estado actual ({estado_actual})")
    largo = proy_markov.melt(id_vars="Mes_horizonte",
                             value_vars=["P_Normal", "P_Vulnerable", "P_Estres",
                                         "P_tocar_Estres_hasta_h"],
                             var_name="Probabilidad", value_name="Valor")
    fig = px.line(largo, x="Mes_horizonte", y="Valor", color="Probabilidad", markers=True,
                  labels={"Mes_horizonte": "Meses hacia adelante"})
    st.plotly_chart(fig, width="stretch")

    with st.expander("Supuestos: ¿es razonable el modelo de Markov?"):
        st.dataframe(tabla("markov_homogeneidad.csv", index_col=0).round(3))
        cal = tabla("markov_calibracion.csv").set_index("Indicador")["Valor"]
        st.markdown(
            f"""
- **Homogeneidad temporal:** la permanencia en Estrés baja de la primera a la segunda mitad
  de la muestra (las crisis recientes fueron más cortas). La matriz no es idéntica en el tiempo.
- **Memoria de un mes:** desde Normal, Markov da {pct(float(cal['P_tocar_Estres_12m_Markov']))}
  de tocar Estrés en 12 meses, frente a {pct(float(cal['Frecuencia_historica_desde_estado_actual']))}
  observado históricamente. La cadena de primer orden subestima ese riesgo porque olvida la historia
  más allá del mes actual.
"""
        )

# ============================================================
# MONTE CARLO
# ============================================================
elif seccion == "Monte Carlo":
    st.title("Simulación Monte Carlo basada en el VAR(2)")
    st.markdown(
        f"""
- **Mecanismo:** cada trayectoria aplica las ecuaciones del VAR(2) durante 12 meses con
  vectores de residuos históricos tomados al azar (*bootstrap*). Así se conservan las colas
  pesadas y la correlación entre variables.
- **Simulaciones:** {int(mc_met['Simulaciones']):,} · **semilla:** {int(mc_met['Semilla'])} ·
  **punto de partida:** Nasdaq = {mc_met['Nasdaq_inicial']:,.0f} ({fecha_actual:%b %Y}).
- Cada mes simulado se clasifica con los **mismos estados de Markov**.
"""
    )
    tarjeta_alerta()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rentabilidad mediana 12m", pct(mc['Ret_acum_p50_pct'].iloc[-1] / 100))
    c2.metric("P(rentabilidad < 0)", pct(mc_met["Prob_rentabilidad_negativa_12m"]))
    c3.metric("VaR 95% 12m", "-" + pct(mc_met['VaR95_12m_pct'] / 100))
    c4.metric("CVaR 95% 12m", "-" + pct(mc_met['CVaR95_12m_pct'] / 100))

    hist = niveles.set_index("Fecha")["NASDAQCOM"].iloc[-36:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist, name="Observado", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=mc["Fecha"], y=mc["Indice_p95"], mode="lines", line=dict(width=0),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=mc["Fecha"], y=mc["Indice_p5"], fill="tonexty", mode="lines",
                             fillcolor="rgba(21,101,192,0.2)", line=dict(width=0),
                             name="Percentil 5-95"))
    fig.add_trace(go.Scatter(x=mc["Fecha"], y=mc["Indice_p50"], name="Mediana", mode="lines",
                             line=dict(color="#1565c0", width=3)))
    fig.update_layout(title="Nasdaq: escenarios simulados a 12 meses", yaxis_title="Índice")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Escenarios a 12 meses")
    st.dataframe(tabla("mc_escenarios_12m.csv").round(1), hide_index=True)

    st.subheader("Probabilidad de cada estado por mes: Monte Carlo vs Markov")
    comp = pd.DataFrame({
        "Mes": mc["Mes_horizonte"],
        "Estrés (Monte Carlo)": mc["P_Estres"],
        "Estrés (Markov)": proy_markov["P_Estres"],
        "Tocar Estrés (Monte Carlo)": mc["P_tocar_Estres_hasta_h"],
        "Tocar Estrés (Markov)": proy_markov["P_tocar_Estres_hasta_h"],
    }).melt(id_vars="Mes", var_name="Serie", value_name="Probabilidad")
    fig = px.line(comp, x="Mes", y="Probabilidad", color="Serie", markers=True)
    fig.add_hline(y=umbral, line_dash="dot", annotation_text="Umbral de alerta")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Monte Carlo da más riesgo que Markov en los primeros meses. El VAR usa una volatilidad "
        "constante (la de toda la muestra), pero la prueba ARCH muestra que la volatilidad del "
        "Nasdaq cambia: en meses tranquilos es menor. Markov sí recoge esa calma porque el estado "
        "Normal es persistente. Las dos cifras funcionan como cota alta y cota baja."
    )

    st.subheader("Explora otro umbral de drawdown")
    tray = tabla("mc_trayectorias.csv.gz", compression="gzip")
    dd = st.slider("Caída máxima frente al pico de 12 meses", 0.05, 0.40, 0.20, 0.05)
    st.metric(f"P(drawdown máximo > {pct(dd, 0)} en 12 meses)",
              pct((tray["DD_max_12m"] > dd).mean()))

    with st.expander("Estabilidad numérica"):
        est = tabla("mc_estabilidad.csv")
        st.dataframe(est.round(4), hide_index=True)
        st.caption("Con 5.000 trayectorias el error estándar de la probabilidad es menor a "
                   "1 punto porcentual.")

# ============================================================
# CONCLUSIONES
# ============================================================
elif seccion == "Conclusiones":
    st.title("Conclusiones")
    tarjeta_alerta()
    st.markdown(
        f"""
### Hallazgos
- **El Nasdaq anticipa a las variables macro, no al revés.** En el VAR(2), sus rezagos
  ayudan a predecir el WTI, el diferencial Baa y la tasa a 10 años (Granger, 5%), pero ninguna
  de las tres anticipa al Nasdaq. Para el Middle Office, esas tres variables no sirven como
  señal temprana del índice; sirven para entender cómo se transmite un choque.
- **El riesgo es persistente.** Una vez en Estrés, el Nasdaq se queda ahí con probabilidad
  {pct(matriz.loc['Estres', 'Estres'])} al mes. Vulnerable es un estado de paso: una alerta
  emitida en Vulnerable llega a tiempo para actuar.
- **Situación actual ({fecha_actual:%b %Y}):** estado {estado_actual}. La probabilidad de estar en
  Estrés dentro de 12 meses va de {pct(proy_markov['P_Estres'].iloc[-1])} (Markov) a
  {pct(mc['P_Estres'].iloc[-1])} (Monte Carlo), frente a una frecuencia histórica de largo
  plazo de 22,6%.

### Decisión
Con la regla elegida ({metrica_alerta.lower()}, h = {horizonte}, umbral {pct(umbral, 0)}),
la alerta está **{'ACTIVA' if alerta else 'INACTIVA'}**.

- **Por qué 30%:** históricamente el Nasdaq ha estado en Estrés el 22,6% de los meses.
  Un umbral de 30% avisa solo cuando el riesgo está claramente por encima de lo habitual.
- **Por qué Monte Carlo y no Markov:** Markov da una cifra más baja
  ({pct(proy_markov['P_Estres'].iloc[-1])}) porque solo recuerda el mes actual. Para un área de
  riesgo se usa la estimación más prudente.

### Límites
- Asociación temporal y causalidad de Granger, no causalidad estructural.
- El VAR supone volatilidad constante; la prueba ARCH lo contradice para Nasdaq y WTI.
- La cadena de Markov es de primer orden y su matriz cambia entre submuestras.
- El WTI es un proxy energético general, no el costo eléctrico de los centros de datos.
- Escenarios, no predicciones: el percentil 5 no es el peor caso posible.

### Fuentes y bibliografía
- Federal Reserve Bank of St. Louis, FRED: NASDAQCOM, GS10, MCOILWTICO, BAA10YM.
- Lütkepohl, H. *New Introduction to Multiple Time Series Analysis*.
- Hyndman, R. J. y Athanasopoulos, G. *Forecasting: Principles and Practice*.
- Ross, S. M. *Introduction to Probability Models*.
- McKinney, W. *Python for Data Analysis*.
- Documentación oficial de pandas, statsmodels, Plotly y Streamlit.
"""
    )
