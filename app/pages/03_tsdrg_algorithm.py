from PIL import Image
from pathlib import Path

import streamlit as st


# Title
st.markdown("# 🌳 Tree Tensor Network Strong Disorder Renormalization Group (tSDRG)")
st.sidebar.markdown(
    "# 🌳 Tree Tensor Network Strong Disorder Renormalization Group (tSDRG)"
)
st.markdown("Here we describe how the algorithm proceed.")

st.header("1. The algorithm")
st.markdown("### 📽️ Tensor tree as a projector")
st.latex(r"\tilde{H} = P^T H P")
st.markdown("with")
st.latex(r"\dim \tilde{H} \ll \dim H")

st.markdown("#")
st.markdown("#")
image = Image.open(Path.cwd() / "app/images/tsdrg_measurements.png")
with st.container():
    st.image(
        image,
        width=400,
        caption="Source: Griffiths singularities in the random quantum Ising "
        "antiferromagnet: A tree tensor network renormalization group study, "
        "Lin, Y.-P., Kao, Y.-J., Chen, P., & Lin, Y.-C. (2017), "
        "Phys. Rev. B, 96, 064427 (2017).",
    )

st.markdown("#")
image = Image.open(Path.cwd() / "app/images/tsdrg_algorithm.png")
with st.container():
    st.image(
        image,
        width=400,
        caption="Source: Griffiths singularities in the random quantum Ising "
        "antiferromagnet: A tree tensor network renormalization group study, "
        "Lin, Y.-P., Kao, Y.-J., Chen, P., & Lin, Y.-C. (2017), "
        "Phys. Rev. B, 96, 064427 (2017).",
    )

st.markdown("## 👣 Steps")
st.markdown(
    r"1. Decompose the chain into a set of n-site blocks and construct the block MPO "
    r"tensors W_B."
)
st.markdown(
    r"2. Obtain the energy spectrum of the two-block Hamiltonian for each pair of "
    r"nearest-neighbor blocks."
)
st.markdown(
    r"3. Merge the pair of blocks (say B and B') with the largest energy gap "
    r"\Delta^{max} ≡ max(\Delta_{\chi}) into a new block and contract the MPO tensors "
    r"to form W_{BB'}."
)
st.markdown(
    r"4. Build a rank-3 isometry tensor V with the \chi lowest energy states of new "
    r"block as column vectors; use V and V^{\dagger} to truncate W_{BB'} to "
    r"W_{\tilde{B}}."
)
st.markdown(r"5. Repeat steps 2–4 until the system is represented by one single block.")
