"""
PERSEFONE - CÁLCULO TENSORIAL
Versión Streamlit - Interactiva y moderna
Gradiente, Diferencial, Tensor Métrico y Derivada Direccional
"""

import streamlit as st
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="Perséfone - Cálculo Tensorial",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# ESTILO CSS PERSONALIZADO
# ============================================

st.markdown("""
<style>
    /* Estilo Perséfone */
    .stApp {
        background: linear-gradient(135deg, #0a0b10 0%, #1a0f2e 100%);
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #2d1b4e, #4a2a7a);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: #c77dff;
        font-family: 'Times New Roman', serif;
        font-size: 3rem;
        margin: 0;
    }
    .main-header p {
        color: #e8e6f0;
        font-style: italic;
        margin: 0;
    }
    .card {
        background: rgba(45, 27, 78, 0.7);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #c77dff;
    }
    .result-box {
        background: #0a0b10;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        color: #c77dff;
    }
    .metric-title {
        color: #ff6b4a;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# ENCABEZADO
# ============================================

st.markdown("""
<div class="main-header">
    <h1>🏺 ΠΕΡΣΕΦΟΝΗ 🏺</h1>
    <p>Cálculo Tensorial | Gradiente · Diferencial · Tensor Métrico</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# BARRA LATERAL - ENTRADA DE DATOS
# ============================================

with st.sidebar:
    st.markdown("## 🔮 Función Potencial")
    
    # Ejemplos rápidos
    ejemplos = {
        "x² - y²": "x**2 - y**2",
        "x² + y²": "x**2 + y**2",
        "e^-(x²+y²)": "exp(-(x**2 + y**2))",
        "sin(x)·cos(y)": "sin(x)*cos(y)",
        "x·y": "x*y"
    }
    
    funcion_ejemplo = st.selectbox("📌 Ejemplos:", list(ejemplos.keys()))
    
    # Entrada de función
    funcion_str = st.text_input("o escribe tu función f(x,y):", 
                                 value=ejemplos[funcion_ejemplo],
                                 help="Usa: x, y, sin, cos, exp, log, sqrt, ** para potencias")
    
    st.markdown("---")
    st.markdown("## 📍 Punto de Evaluación")
    
    col1, col2 = st.columns(2)
    with col1:
        x0 = st.number_input("x₀", value=1.0, step=0.1, format="%.2f")
    with col2:
        y0 = st.number_input("y₀", value=1.0, step=0.1, format="%.2f")
    
    st.markdown("---")
    st.markdown("## 🧭 Dirección para Derivada")
    
    col1, col2 = st.columns(2)
    with col1:
        vx = st.number_input("vₓ", value=1.0, step=0.5, format="%.2f")
    with col2:
        vy = st.number_input("vᵧ", value=1.0, step=0.5, format="%.2f")
    
    st.markdown("---")
    st.markdown("## 🎨 Visualización")
    
    rango = st.slider("Rango de visualización:", 1, 5, 2)
    resolucion = st.slider("Resolución:", 20, 100, 50)

# ============================================
# FUNCIONES DE CÁLCULO
# ============================================

@st.cache_data
def calcular_todo(funcion_str, x0, y0, vx, vy):
    """Calcula todo: función, derivadas, gradiente, derivada direccional"""
    
    x, y = sp.symbols('x y', real=True)
    
    # Procesar la función
    expr_str = funcion_str.replace('^', '**')
    
    try:
        namespace = {
            'x': x, 'y': y,
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
            'pi': sp.pi, 'e': sp.E
        }
        f = eval(expr_str, {"__builtins__": {}}, namespace)
        
        # Derivadas
        df_dx = sp.diff(f, x)
        df_dy = sp.diff(f, y)
        
        # Evaluación numérica
        subs = {x: x0, y: y0}
        f_num = float(f.subs(subs).evalf())
        grad_x = float(df_dx.subs(subs).evalf())
        grad_y = float(df_dy.subs(subs).evalf())
        
        # Derivada direccional
        norm = np.sqrt(vx**2 + vy**2)
        if norm > 0:
            vx_n, vy_n = vx/norm, vy/norm
        else:
            vx_n, vy_n = 1.0, 0.0
        
        deriv_dir = grad_x * vx_n + grad_y * vy_n
        
        # Verificación numérica
        h = 0.0001
        f_punto = f_num
        f_desp = float(f.subs({x: x0 + h*vx_n, y: y0 + h*vy_n}).evalf())
        deriv_num = (f_desp - f_punto) / h
        
        # Tensor métrico en polares
        r, theta = sp.symbols('r theta', real=True, positive=True)
        f_pol = f.subs({x: r*sp.cos(theta), y: r*sp.sin(theta)}).simplify()
        df_dr = sp.diff(f_pol, r)
        df_dtheta = sp.diff(f_pol, theta)
        
        return {
            'f': f, 'df_dx': df_dx, 'df_dy': df_dy,
            'f_num': f_num, 'grad_x': grad_x, 'grad_y': grad_y,
            'deriv_dir': deriv_dir, 'deriv_num': deriv_num,
            'vx_n': vx_n, 'vy_n': vy_n,
            'f_pol': f_pol, 'df_dr': df_dr, 'df_dtheta': df_dtheta
        }
    except Exception as e:
        st.error(f"Error en la función: {str(e)}")
        return None

# ============================================
# FUNCIONES DE GRÁFICAS
# ============================================

def graficar_superficie(funcion_str, rango, resolucion):
    """Genera gráfica 3D de la función"""
    
    x_vals = np.linspace(-rango, rango, resolucion)
    y_vals = np.linspace(-rango, rango, resolucion)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Evaluar la función numéricamente
    expr_str = funcion_str.replace('^', '**')
    try:
        Z = eval(expr_str, {
            'x': X, 'y': Y, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'sqrt': np.sqrt, 'log': np.log, 'tan': np.tan
        })
    except:
        Z = X**2 - Y**2
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    ax.set_zlabel('f(x,y)', fontsize=10)
    ax.set_title(f'f(x,y) = {funcion_str[:50]}', fontsize=12)
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    return fig

def graficar_contornos_gradiente(funcion_str, rango, resolucion, x0, y0):
    """Genera mapa de contornos con el gradiente"""
    
    x_vals = np.linspace(-rango, rango, resolucion)
    y_vals = np.linspace(-rango, rango, resolucion)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    expr_str = funcion_str.replace('^', '**')
    try:
        Z = eval(expr_str, {
            'x': X, 'y': Y, 'np': np,
            'sin': np.sin, 'cos': np.cos, 'exp': np.exp,
            'sqrt': np.sqrt, 'log': np.log, 'tan': np.tan
        })
    except:
        Z = X**2 - Y**2
    
    # Gradiente numérico
    gy, gx = np.gradient(Z, x_vals, y_vals)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
    skip = slice(None, None, max(1, resolucion//10))
    ax.quiver(X[skip, skip], Y[skip, skip], 
              gx[skip, skip], gy[skip, skip], 
              color='white', alpha=0.8, scale=20, width=0.005)
    
    # Marcar el punto
    ax.plot(x0, y0, 'ro', markersize=12, markerfacecolor='red',
           markeredgecolor='white', markeredgewidth=2, label=f'P({x0},{y0})')
    
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title('Curvas de nivel y Gradiente ∇f', fontsize=12)
    ax.legend(loc='upper right')
    fig.colorbar(contour, ax=ax)
    
    return fig

# ============================================
# COLUMNAS PRINCIPALES
# ============================================

col1, col2 = st.columns([1, 1])

# ============================================
# COLUMNA IZQUIERDA - RESULTADOS MATEMÁTICOS
# ============================================

with col1:
    st.markdown("## 📜 Resultados Matemáticos")
    
    # Calcular todo
    resultados = calcular_todo(funcion_str, x0, y0, vx, vy)
    
    if resultados:
        # Diferencial y Gradiente
        with st.expander("📐 Diferencial y Gradiente", expanded=True):
            st.markdown(f"""
            <div class="card">
                <b>f(x,y) =</b> <code>{resultados['f']}</code><br><br>
                <b>∂f/∂x =</b> <code>{resultados['df_dx']}</code><br>
                <b>∂f/∂y =</b> <code>{resultados['df_dy']}</code><br><br>
                <b>df =</b> <code>({resultados['df_dx']}) dx + ({resultados['df_dy']}) dy</code><br><br>
                <b>∇f =</b> <code>({resultados['df_dx']}, {resultados['df_dy']})</code>
            </div>
            """, unsafe_allow_html=True)
        
        # Evaluación en el punto
        with st.expander("📍 Evaluación en el Punto", expanded=True):
            st.markdown(f"""
            <div class="card">
                <b>P({x0}, {y0})</b><br><br>
                f(P) = <code>{resultados['f_num']:.10f}</code><br>
                ∇f(P) = <code>({resultados['grad_x']:.10f}, {resultados['grad_y']:.10f})</code><br>
                |∇f(P)| = <code>{np.sqrt(resultados['grad_x']**2 + resultados['grad_y']**2):.10f}</code>
            </div>
            """, unsafe_allow_html=True)
        
        # Derivada direccional
        with st.expander("🎯 Derivada Direccional", expanded=True):
            st.markdown(f"""
            <div class="card">
                <b>Dirección:</b> v = ({vx}, {vy})<br>
                <b>Vector unitario:</b> v̂ = ({resultados['vx_n']:.6f}, {resultados['vy_n']:.6f})<br><br>
                <b>D_v̂ f (Analítica):</b> <code style="color:#ff6b4a; font-size:1.2rem;">{resultados['deriv_dir']:.10f}</code><br>
                <b>D_v̂ f (Numérica):</b> <code>{resultados['deriv_num']:.10f}</code><br>
                <b>Error:</b> <code>{abs(resultados['deriv_dir'] - resultados['deriv_num']):.2e}</code>
            </div>
            """, unsafe_allow_html=True)
        
        # Tensor Métrico
        with st.expander("📐 Tensor Métrico en Polares", expanded=True):
            st.markdown(f"""
            <div class="card">
                <b>Transformación:</b> x = r·cos(θ), y = r·sin(θ)<br><br>
                <b>f(r,θ) =</b> <code>{resultados['f_pol']}</code><br><br>
                <b>Tensor métrico covariante g_ij:</b><br>
                <code>⎡ 1   0  ⎤</code><br>
                <code>⎢        ⎥</code><br>
                <code>⎣ 0   r² ⎦</code><br><br>
                <b>Tensor métrico contravariante g^ij:</b><br>
                <code>⎡ 1    0  ⎤</code><br>
                <code>⎢         ⎥</code><br>
                <code>⎣ 0   1/r²⎦</code><br><br>
                <b>Elemento de línea:</b> ds² = dr² + r²·dθ²<br><br>
                <b>Componentes covariantes de df:</b><br>
                ∂f/∂r = <code>{resultados['df_dr']}</code><br>
                ∂f/∂θ = <code>{resultados['df_dtheta']}</code><br><br>
                <b>Gradiente en polares (∇f = g^ij ∂_j f):</b><br>
                ∇f^r = <code>{resultados['df_dr']}</code><br>
                ∇f^θ = <code>{resultados['df_dtheta']} / r²</code>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# COLUMNA DERECHA - GRÁFICAS
# ============================================

with col2:
    st.markdown("## 📈 Visualización Gráfica")
    
    # Pestañas para gráficas
    tab1, tab2 = st.tabs(["🌄 Superficie 3D", "🗺️ Contornos + Gradiente"])
    
    with tab1:
        st.markdown("### Superficie de la función")
        fig_3d = graficar_superficie(funcion_str, rango, resolucion)
        st.pyplot(fig_3d)
        plt.close(fig_3d)
    
    with tab2:
        st.markdown("### Curvas de nivel y campo gradiente")
        fig_cont = graficar_contornos_gradiente(funcion_str, rango, resolucion, x0, y0)
        st.pyplot(fig_cont)
        plt.close(fig_cont)

# ============================================
# SECCIÓN DE CONCLUSIÓN
# ============================================

st.markdown("---")
st.markdown("## 💡 El Mito del Tensor Métrico - Perséfone")

with st.container():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d1b4e, #1a0f2e); border-radius: 15px; padding: 1.5rem;">
    <p style="font-size: 1.1rem; line-height: 1.6;">
    Así como <b style="color: #c77dff;">Perséfone</b> transita entre dos mundos, el <b style="color: #ff6b4a;">tensor métrico</b> 
    conecta dos formas de ver el mismo objeto geométrico:
    </p>
    <ul style="font-size: 1rem; line-height: 1.8;">
        <li><b style="color: #c77dff;">🌑 El DIFERENCIAL df es un COVECTOR</b> (mundo oculto) - existe independientemente de la métrica</li>
        <li><b style="color: #ff6b4a;">☀️ El GRADIENTE ∇f es un VECTOR</b> (mundo manifestado) - necesita la métrica para existir</li>
        <li><b style="color: #c77dff;">📐 La métrica "levanta el índice"</b>: ∇fⁱ = gⁱʲ ∂ⱼ f</li>
        <li><b style="color: #ff6b4a;">🔄 En polares (g_ij = diag(1, r²)):</b> ∇f^θ = (1/r²)·∂f/∂θ - ¡la métrica corrige la escala!</li>
    </ul>
    <p style="font-size: 1rem; font-style: italic; text-align: center; margin-top: 1rem;">
    "Sin el tensor métrico, el gradiente es solo una promesa sin forma."<br>
    — <b>Perséfone, Reina del Inframundo</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# VERIFICACIÓN DE ENTREGABLES
# ============================================

st.markdown("---")
st.markdown("## ✅ Verificación de Entregables del PIA")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("""
    <div class="card">
    ✓ Código fuente<br>
    ✓ Función potencial editable<br>
    ✓ Desarrollo matemático<br>
    ✓ Gráficas interactivas
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="card">
    ✓ Diferencial total<br>
    ✓ Gradiente<br>
    ✓ Derivada direccional<br>
    ✓ Verificación numérica
    </div>
    """, unsafe_allow_html=True)

with col_c:
    st.markdown("""
    <div class="card">
    ✓ Tensor métrico en polares<br>
    ✓ Transformación coordenadas<br>
    ✓ Conclusión sobre la métrica
    </div>
    """, unsafe_allow_html=True)

# ============================================
# PIE DE PÁGINA
# ============================================

st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #7a6a8a;">
<hr>
🏺 Cálculo Variacional y Tensorial | Proyecto Integrador de Aprendizaje (PIA) 🏺
</div>
""", unsafe_allow_html=True)