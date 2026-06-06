"""
Asignación Óptima de Mezcla de Productos
Modelo de Markowitz aplicado a portafolio comercial
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Asignación Óptima de Mezcla de Productos",
    layout="wide",
    page_icon="📊",
)

# ─────────────────────────────────────────────────────────────
# Título e introducción
# ─────────────────────────────────────────────────────────────
st.title("📊 Asignación Óptima de Mezcla de Productos")
st.markdown(
    """
    ### Modelo de Markowitz Aplicado al Portafolio Comercial

    Esta aplicación adapta la **Teoría Moderna de Portafolios de Harry Markowitz** al mundo
    comercial: en lugar de acciones bursátiles, los **"activos"** son tus productos o servicios.

    El modelo responde la pregunta clave:

    > **¿Cuál es la combinación óptima de productos para maximizar la rentabilidad
    > sobre un monto de ventas definido?**

    **Lógica del modelo:**
    - **Rendimiento esperado** → ROI histórico promedio de cada producto (crecimiento año a año)
    - **Riesgo** → Desviación estándar histórica del ROI comercial
    - **Diversificación** → Matriz de covarianza entre los ROI de los productos
    - **Frontera Eficiente** → Combinaciones óptimas riesgo-rendimiento (simulación Monte Carlo)
    - **Máximo Sharpe** → La cartera que mejor compensa el riesgo asumido

    Carga tu archivo Excel para comenzar el análisis.
    """
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    archivo = st.file_uploader(
        "Carga tu archivo Excel (.xlsx)",
        type=["xlsx"],
        help="Primera columna: nombre del producto. Columnas siguientes: ventas históricas por período.",
    )

    st.markdown("---")
    n_simulaciones = st.slider(
        "Número de simulaciones Monte Carlo",
        min_value=500,
        max_value=10000,
        value=3000,
        step=500,
        help="Más simulaciones = mayor precisión, mayor tiempo de cómputo.",
    )

    monto_total = st.number_input(
        "Monto total de ventas a distribuir ($)",
        min_value=1000.0,
        value=1_000_000.0,
        step=10_000.0,
        format="%.2f",
        help="Ingresa el presupuesto comercial que deseas optimizar.",
    )

    tasa_libre_riesgo = st.number_input(
        "Tasa libre de riesgo (%)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Usada para calcular el índice de Sharpe (ej. CETES, bono gubernamental).",
    ) / 100.0

    st.markdown("---")
    st.caption("Modelo de Markowitz · Portafolio Comercial")

# ─────────────────────────────────────────────────────────────
# Mensaje inicial si no hay archivo
# ─────────────────────────────────────────────────────────────
if archivo is None:
    st.info(
        "👈 **Carga un archivo Excel** en el panel lateral para iniciar el análisis.\n\n"
        "**Formato esperado:**\n"
        "- Primera columna: nombre del producto/servicio\n"
        "- Columnas siguientes: ventas históricas por período (año, mes, etc.)\n"
        "- Al menos **3 períodos históricos** por producto"
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# Carga y validación de datos
# ─────────────────────────────────────────────────────────────
try:
    # Intentar leer el Excel detectando si hay encabezados adicionales
    df_raw = pd.read_excel(archivo, header=None)

    # Encontrar la fila que contiene los encabezados (primera columna no numérica con varios valores)
    header_row = None
    for i, row in df_raw.iterrows():
        vals = row.dropna().tolist()
        if len(vals) >= 2 and isinstance(vals[0], str) and not str(vals[0]).startswith("*"):
            # Verificar que las otras columnas sean años o períodos (string/int)
            non_null = row.dropna()
            if len(non_null) >= 3:
                header_row = i
                break

    if header_row is None:
        st.error("No se encontró una fila de encabezados válida en el archivo.")
        st.stop()

    df = pd.read_excel(archivo, header=header_row)

    # Renombrar primera columna a "Producto"
    primera_col = df.columns[0]
    df = df.rename(columns={primera_col: "Producto"})

    # Eliminar filas sin producto
    df = df[df["Producto"].notna()]
    df = df[~df["Producto"].astype(str).str.startswith("*")]  # quitar notas

    # Detectar columnas de ventas (excluir columnas como "Total" o similares)
    cols_datos = [
        c for c in df.columns
        if c != "Producto" and str(c).upper() not in ["TOTAL", "TOTAL ACUMULADO", "ACUMULADO"]
    ]

    # Filtrar filas que son totales
    df = df[~df["Producto"].astype(str).str.upper().str.contains("TOTAL")]

    # Convertir columnas de datos a numérico
    for col in cols_datos:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar filas con datos faltantes en columnas de ventas
    df = df.dropna(subset=cols_datos)
    df = df.reset_index(drop=True)

    # Validaciones
    if len(df) < 2:
        st.error("Se necesitan al menos **2 productos** para construir un portafolio.")
        st.stop()

    if len(cols_datos) < 3:
        st.error("Se necesitan al menos **3 períodos históricos** para calcular estadísticas.")
        st.stop()

    # Tabla de ventas
    df_ventas = df[["Producto"] + cols_datos].set_index("Producto")

    # ─────────────────────────────────────────────────────────
    # Cálculo de ROI período a período
    # ─────────────────────────────────────────────────────────
    # ROI_t = (Ventas_t - Ventas_{t-1}) / Ventas_{t-1}
    df_roi = df_ventas.T.pct_change().dropna()  # períodos como filas, productos como columnas

    if len(df_roi) < 2:
        st.error("Se necesitan al menos **3 períodos** para calcular retornos. Solo se detectaron suficientes datos para 1 retorno.")
        st.stop()

    productos = df_ventas.index.tolist()
    n_productos = len(productos)

    # ─────────────────────────────────────────────────────────
    # Estadísticas base
    # ─────────────────────────────────────────────────────────
    rendimientos_esperados = df_roi.mean()       # ROI promedio por producto
    riesgos = df_roi.std()                       # Desv. estándar del ROI
    matriz_covarianza = df_roi.cov()

    # ─────────────────────────────────────────────────────────
    # Sección 1: Datos históricos de ventas
    # ─────────────────────────────────────────────────────────
    st.header("1️⃣ Base de Datos Histórica de Ventas")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Ventas por Producto y Período")
        st.dataframe(
            df_ventas.style.format("${:,.0f}"),
            use_container_width=True,
        )
    with col2:
        st.subheader("ROI por Período")
        st.dataframe(
            df_roi.style.format("{:.2%}"),
            use_container_width=True,
        )

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Sección 2: Estadísticas individuales
    # ─────────────────────────────────────────────────────────
    st.header("2️⃣ Rendimiento y Riesgo por Producto")

    df_stats = pd.DataFrame({
        "Rendimiento Esperado (ROI)": rendimientos_esperados,
        "Riesgo (Desv. Estándar)": riesgos,
        "Sharpe Individual": (rendimientos_esperados - tasa_libre_riesgo) / riesgos,
    })

    st.dataframe(
        df_stats.style.format({
            "Rendimiento Esperado (ROI)": "{:.2%}",
            "Riesgo (Desv. Estándar)": "{:.2%}",
            "Sharpe Individual": "{:.4f}",
        }),
        use_container_width=True,
    )

    # Gráfica de barras: rendimiento vs riesgo
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        name="Rendimiento Esperado",
        x=productos,
        y=rendimientos_esperados.values * 100,
        marker_color="#2196F3",
    ))
    fig_bar.add_trace(go.Bar(
        name="Riesgo (Desv. Estándar)",
        x=productos,
        y=riesgos.values * 100,
        marker_color="#F44336",
    ))
    fig_bar.update_layout(
        title="Rendimiento vs Riesgo por Producto (%)",
        barmode="group",
        yaxis_title="(%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Sección 3: Matriz de Covarianza
    # ─────────────────────────────────────────────────────────
    st.header("3️⃣ Matriz de Covarianza entre Productos")
    st.markdown(
        "La **covarianza** captura cómo se mueven conjuntamente los ROI de los productos. "
        "Es la base matemática para calcular el riesgo del portafolio y encontrar combinaciones diversificadas."
    )

    col_cov_tab, col_cov_heat = st.columns(2)

    with col_cov_tab:
        st.subheader("Tabla de Covarianza")
        st.dataframe(
            matriz_covarianza.style.format("{:.6f}"),
            use_container_width=True,
        )

    with col_cov_heat:
        fig_cov = px.imshow(
            matriz_covarianza,
            text_auto=".4f",
            color_continuous_scale="Blues",
            title="Heatmap de Covarianza",
            aspect="auto",
        )
        fig_cov.update_layout(template="plotly_white")
        st.plotly_chart(fig_cov, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Sección 4: Simulación Monte Carlo y Frontera Eficiente
    # ─────────────────────────────────────────────────────────
    st.header("4️⃣ Simulación Monte Carlo — Frontera Eficiente Comercial")
    st.markdown(
        f"Se simulan **{n_simulaciones:,}** combinaciones aleatorias de pesos para los {n_productos} productos. "
        "Cada punto representa un portafolio posible. La curva resultante es la **Frontera Eficiente Comercial**."
    )

    cov_matrix = matriz_covarianza.values
    ret_esperados = rendimientos_esperados.values

    # Arrays de resultados
    sim_pesos = np.zeros((n_simulaciones, n_productos))
    sim_rendimiento = np.zeros(n_simulaciones)
    sim_riesgo = np.zeros(n_simulaciones)
    sim_sharpe = np.zeros(n_simulaciones)

    np.random.seed(42)
    for i in range(n_simulaciones):
        pesos = np.random.random(n_productos)
        pesos /= pesos.sum()  # normalizar a 1

        ret = np.dot(pesos, ret_esperados)
        riesgo = np.sqrt(np.dot(pesos, np.dot(cov_matrix, pesos)))
        sharpe = (ret - tasa_libre_riesgo) / riesgo if riesgo > 0 else 0

        sim_pesos[i] = pesos
        sim_rendimiento[i] = ret
        sim_riesgo[i] = riesgo
        sim_sharpe[i] = sharpe

    # Portafolio óptimo (máximo Sharpe)
    idx_optimo = np.argmax(sim_sharpe)
    pesos_optimos = sim_pesos[idx_optimo]
    ret_optimo = sim_rendimiento[idx_optimo]
    riesgo_optimo = sim_riesgo[idx_optimo]
    sharpe_optimo = sim_sharpe[idx_optimo]

    # Gráfica de Frontera Eficiente
    df_sim = pd.DataFrame({
        "Riesgo (%)": sim_riesgo * 100,
        "Rendimiento (%)": sim_rendimiento * 100,
        "Sharpe": sim_sharpe,
    })

    fig_frontera = go.Figure()

    # Todos los portafolios
    fig_frontera.add_trace(go.Scatter(
        x=df_sim["Riesgo (%)"],
        y=df_sim["Rendimiento (%)"],
        mode="markers",
        marker=dict(
            size=4,
            color=df_sim["Sharpe"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Sharpe"),
            opacity=0.7,
        ),
        name="Portafolios simulados",
        hovertemplate=(
            "Riesgo: %{x:.2f}%<br>"
            "Rendimiento: %{y:.2f}%<br>"
            "Sharpe: %{marker.color:.4f}<extra></extra>"
        ),
    ))

    # Portafolio óptimo (estrella verde)
    fig_frontera.add_trace(go.Scatter(
        x=[riesgo_optimo * 100],
        y=[ret_optimo * 100],
        mode="markers",
        marker=dict(
            symbol="star",
            size=20,
            color="lime",
            line=dict(color="darkgreen", width=1.5),
        ),
        name="⭐ Óptimo (Máx. Sharpe)",
        hovertemplate=(
            f"<b>Portafolio Óptimo</b><br>"
            f"Riesgo: {riesgo_optimo*100:.2f}%<br>"
            f"Rendimiento: {ret_optimo*100:.2f}%<br>"
            f"Sharpe: {sharpe_optimo:.4f}<extra></extra>"
        ),
    ))

    fig_frontera.update_layout(
        title="Frontera Eficiente Comercial — Portafolio de Productos",
        xaxis_title="Riesgo — Desviación Estándar del ROI (%)",
        yaxis_title="Rendimiento Esperado del ROI (%)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=550,
    )

    st.plotly_chart(fig_frontera, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Sección 5: Portafolio Óptimo
    # ─────────────────────────────────────────────────────────
    st.header("5️⃣ Portafolio Óptimo — Máximo Índice de Sharpe")

    # Métricas
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric(
        label="📈 Rendimiento Esperado",
        value=f"{ret_optimo*100:.2f}%",
        help="ROI esperado del portafolio óptimo",
    )
    col_m2.metric(
        label="📉 Riesgo del Portafolio",
        value=f"{riesgo_optimo*100:.2f}%",
        help="Desviación estándar del ROI del portafolio óptimo",
    )
    col_m3.metric(
        label="⚡ Índice de Sharpe",
        value=f"{sharpe_optimo:.4f}",
        help="Rendimiento ajustado por riesgo (mayor = mejor)",
    )

    st.markdown("---")

    # Asignación óptima
    df_asignacion = pd.DataFrame({
        "Producto": productos,
        "Peso Óptimo (%)": pesos_optimos * 100,
        "Monto Asignado ($)": pesos_optimos * monto_total,
    })
    df_asignacion = df_asignacion.sort_values("Peso Óptimo (%)", ascending=False).reset_index(drop=True)

    col_tabla, col_dona = st.columns([1, 1])

    with col_tabla:
        st.subheader("Asignación por Producto")
        st.dataframe(
            df_asignacion.style.format({
                "Peso Óptimo (%)": "{:.2f}%",
                "Monto Asignado ($)": "${:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            f"**Total a distribuir:** ${monto_total:,.2f}  \n"
            f"**Tasa libre de riesgo:** {tasa_libre_riesgo*100:.2f}%"
        )

    with col_dona:
        st.subheader("Distribución Óptima")
        fig_dona = go.Figure(go.Pie(
            labels=df_asignacion["Producto"],
            values=df_asignacion["Peso Óptimo (%)"],
            hole=0.45,
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Peso: %{value:.2f}%<br>"
                "Monto: $%{customdata:,.2f}<extra></extra>"
            ),
            customdata=df_asignacion["Monto Asignado ($)"],
        ))
        fig_dona.update_layout(
            title="Proporción Óptima de Productos",
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="v"),
            height=400,
        )
        st.plotly_chart(fig_dona, use_container_width=True)

    st.divider()

    # ─────────────────────────────────────────────────────────
    # Sección 6: Comparativo productos vs portafolio óptimo
    # ─────────────────────────────────────────────────────────
    st.header("6️⃣ Mapa Riesgo-Rendimiento: Productos vs Portafolio Óptimo")

    fig_comp = go.Figure()

    # Productos individuales
    for prod in productos:
        fig_comp.add_trace(go.Scatter(
            x=[riesgos[prod] * 100],
            y=[rendimientos_esperados[prod] * 100],
            mode="markers+text",
            marker=dict(size=12, symbol="circle"),
            text=[prod],
            textposition="top center",
            name=prod,
            hovertemplate=(
                f"<b>{prod}</b><br>"
                f"Riesgo: {riesgos[prod]*100:.2f}%<br>"
                f"Rendimiento: {rendimientos_esperados[prod]*100:.2f}%<extra></extra>"
            ),
        ))

    # Portafolio óptimo
    fig_comp.add_trace(go.Scatter(
        x=[riesgo_optimo * 100],
        y=[ret_optimo * 100],
        mode="markers+text",
        marker=dict(size=18, symbol="star", color="lime", line=dict(color="darkgreen", width=1.5)),
        text=["⭐ Portafolio Óptimo"],
        textposition="bottom center",
        name="Portafolio Óptimo",
        hovertemplate=(
            "<b>Portafolio Óptimo</b><br>"
            f"Riesgo: {riesgo_optimo*100:.2f}%<br>"
            f"Rendimiento: {ret_optimo*100:.2f}%<br>"
            f"Sharpe: {sharpe_optimo:.4f}<extra></extra>"
        ),
    ))

    fig_comp.update_layout(
        title="Mapa Riesgo-Rendimiento: Productos vs Portafolio Óptimo",
        xaxis_title="Riesgo (%)",
        yaxis_title="Rendimiento Esperado (%)",
        template="plotly_white",
        height=500,
        showlegend=True,
    )

    st.plotly_chart(fig_comp, use_container_width=True)

    st.caption(
        "⚠️ **Nota metodológica:** El rendimiento se calcula como el cambio porcentual (ROI) entre períodos. "
        "El riesgo es la desviación estándar de esos ROI históricos. "
        "El Índice de Sharpe mide cuánto rendimiento extra se obtiene por unidad de riesgo asumida. "
        "La asignación óptima minimiza el riesgo global y maximiza el Sharpe del portafolio."
    )

except Exception as e:
    st.error(f"❌ Error al procesar el archivo: {e}")
    st.exception(e)
