import pandas as pd
from PIL import Image
from pathlib import Path

import streamlit as st
import plotly.io as pio


# Global streamlit settings
st.set_page_config(layout="wide")
# run_on = 'prod' if os.getenv('USER') == 'appuser' else 'local'
pio.renderers.default = "iframe"

# Title
st.markdown("# 📖 Overview")
st.sidebar.markdown("# 📖 Overview")
st.markdown("This is a Tensor Network study to Many-body Localization.")

# Questions
st.header("1. Questions we want to ask")
st.markdown(
    "* How a generic interacting quantum system developing or avoiding " "ergodicity?"
)
st.markdown("* What is the nature of the MBL transition?")
st.markdown("* What is the role of disorder in MBL transition?")

# Quantum avalanche theory
st.header("2. Quantum avalanche theory")
image = Image.open(Path.cwd() / "app/images/quantum_avalanche.png")
with st.container():
    st.image(
        image,
        width=500,
        caption="Source: Kosterlitz-Thouless scaling at many-body localization phase "
        "transitions, Philipp T. Dumitrescu, Anna Goremykina, "
        "Siddharth A. Parameswaran, Maksym Serbyn, Romain Vasseur, "
        "arXiv:1811.03103 (2018).",
    )

# Bottleneck
st.header("3. Bottleneck")
st.markdown(
    "Currently we don't any reliable numerical tools that can tackle the "
    "interior spectrum of large Hilbert space, nor the ability to simulate "
    "long time evolution."
)

# Hamiltonian
st.header("4. The toy model")
st.write("Heisenberg model with random transversed field")
st.latex(
    r"""
        H = \sum_{i=1}^{N} \mathbf{S}_i\cdot\mathbf{S}_{i+1} + h_i S_i^z
    """
)
st.write("where h_{i} is sampled uniformly within the interval")
st.latex(r"h_{i} \in [-h, h)")

# Level Statistics
st.header("5. Level statistics")

st.subheader("🔹 Gap ratio parameter (r-value)")
st.write("Definition")
st.latex(
    r"""
        r_{n}
        \equiv \frac{\min(\delta_{n}, \delta_{n-1})}{\max(\delta_{n}, \delta_{n-1})}
        = \min\left(
            \frac{\delta_{n}}{\delta_{n+1}}, \frac{\delta_{n+1}}{\delta_{n}}
        \right)
    """
)
st.markdown("where")
st.latex(r"\delta_{n} = E_{n+1} - E_{n}")
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
st.markdown("#")
st.caption("**Note**: Ergodic phase is also referred to as extended phase.")
st.markdown("#")

st.subheader("🔹 Kullback-Leibler divergence")
st.write("Definition")
st.latex(
    r"""
        KL = \sum_{i=1}^{\dim\mathcal{H}} p_i \ln(p_i/q_i)
    """
)
st.markdown("where")
st.latex(r"p_i = |\langle i|n \rangle|^2, \,\, q_i = |\langle i|n' \rangle|^2")
st.markdown("#")

st.subheader("🔹 Participation entropies")
st.write("Definition")
st.latex(
    r"""
        S^P_q(|n\rangle) = \frac{1}{1-q}\ln\sum_{i} p^q_i
    """
)
st.latex(
    r"""
        S^P_1(|n\rangle) = -\sum_{i}p_i \ln p_i
    """
)
st.markdown("#")

st.table(
    pd.DataFrame.from_dict(
        {
            "Ergodic phase": [
                0.5307,
                r"KL_{GOE} = 2",
                "Volume law",
                r"S^P_q = a_q \ln(\dim H)",
            ],
            "Localized phase": [
                0.3863,
                r"KL_{Poisson} \sim \ln(\dim H)",
                "Area law",
                r"S^P_q = l_q \ln(\ln\dim H)",
            ],
        },
        orient="index",
        columns=[
            "r-value",
            "Kullback-Leibler divergence",
            "Entanglement entropy",
            "Participation entropies",
        ],
    )
)


# Some known results
st.header("6. Some known results")
image = Image.open(Path.cwd() / "app/images/mbl-participation_coefficient.png")
with st.container():
    st.image(
        image,
        width=500,
        caption="Source: Many-body localization edge in the random-field Heisenberg "
        "chain, David J. Luitz, Nicolas Laflorencie, and Fabien Alet, "
        "arXiv: 1411.0660 (2014).",
    )
