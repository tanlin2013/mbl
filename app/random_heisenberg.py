import streamlit as st
import plotly.io as pio


# Global streamlit settings
st.set_page_config(layout="wide")
# run_on = 'prod' if os.getenv('USER') == 'appuser' else 'local'
pio.renderers.default = "iframe"

# Title
st.markdown("# 📖 Overview")
st.sidebar.markdown("# 📖 Overview")

# Hamiltonian
st.header("1. The Model")
st.write("Heisenberg model with random transversed field")
st.latex(
    r"""
        H = \sum_{i=1}^{N} \mathbf{S}_i\cdot\mathbf{S}_{i+1} + h_i S_i^z
    """
)
st.write("where h_{i} is sampled uniformly within the interval")
st.latex(r"h_{i} \in [-h, h)")

# Level Statistics
st.header("2. Level Statistics")

st.subheader("Gap ratio parameter (r-value)")
st.write("Definition")
st.latex(
    r"""
        r_{n} = \frac{min(\delta_{n}, \delta_{n-1})}{max(\delta_{n}, \delta_{n-1})}
    """
)
st.markdown("#")

st.markdown("* **Ergodic phase**")
st.write("Theoretical value")
st.latex(
    r"""
        \langle r \rangle = 4 - 2 \sqrt{3} \approx 0.53589(8)
    """
)
st.markdown("* **Localized phase**")
st.write("Theoretical value")
st.latex(
    r"""
        \langle r \rangle = 2 \log{2} - 1 \approx 0.38629(4)
    """
)
