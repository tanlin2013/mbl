import streamlit as st


# Title
st.markdown("# ⏳ Looking up energy window by tSDRG with spectral folding")
st.sidebar.markdown("# ⏳ Looking up energy window by tSDRG with spectral folding")

# Sidebar choices
table = st.sidebar.radio("Database table", ("ed", "tsdrg"))
options_n = (8, 10, 12) if table == "ed" else (8, 10, 12, 14, 16, 18, 20)
n = st.sidebar.radio("System size n", options_n)
options_h = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0)
h = st.sidebar.radio("Disorder strength h", options_h)
options_chi = (2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6)
chi = st.sidebar.radio("Truncation dim chi", options_chi) if table == "tsdrg" else None
relative_offset = (
    st.sidebar.radio("Relative offset", (0.1, 0.5, 0.9)) if table == "tsdrg" else None
)
total_sz = st.sidebar.radio("Restrict Total Sz in", (None, 0, 1))
options_n_conf = (8, 10, 20, 30, 40, 50, None)
n_conf = st.sidebar.radio(
    "Number of disorder trials for making plots (The larger, the slower. None means using all samples)",
    options_n_conf,
)

# Load the data
st.header("1. Data Table")
st.caption("Please wait, it may take a while to load the data.")
df = load_data(n, h, chi=chi, relative_offset=relative_offset, total_sz=total_sz)
st.dataframe(df)

# Level statistics
st.header("2. Gap ratio parameter (r-value)")
r1 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.LEVEL_FIRST)
st.write(f"(Level-then-disorder) averaged `r = {r1}`")
r2 = LevelStatistic.averaged_gap_ratio(df, AverageOrder.DISORDER_FIRST)
st.write(f"(Disorder-then-level) averaged `r = {r2}`")
st.write(f"Relative difference is `{abs(r1 - r2)/max(r1, r2) * 100} %`")
st.write("**Note**: Theoretical value is")
st.write("* Ergodic phase: ~`0.5307`.")
st.write("* Localized phase: ~`0.3863`.")

# Data visualization
st.header("### 3. Visualization")
st.markdown("#### 3.a. Interactive")
st.write("**Hint**: Change parameters on the sidebar.")
plot_pairs = [
    (Columns.en, Columns.level_id),
    (Columns.gap_ratio, Columns.en),
    (Columns.en, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.edge_entropy),
    (Columns.gap_ratio, Columns.bipartite_entropy),
    (Columns.en, Columns.energy_gap),
    (Columns.en, Columns.total_sz),
    (Columns.gap_ratio, Columns.total_sz),
]

if n_conf is not None:
    grouped = df.groupby([Columns.seed])
    mini_df = pd.concat([grouped.get_group(g) for g in list(grouped.groups)[:n_conf]])
else:
    mini_df = df.copy()

for k, v in zip(count(start=1), plot_pairs):
    if not (Columns.total_sz in v and total_sz is not None):
        try:
            if table == "ed":
                st.write(density_histogram_2d(mini_df, v[0], v[1], f"Fig. {k}:"))
            elif table == "tsdrg":
                st.write(scatter_with_error_bar(mini_df, v[0], v[1], f"Fig. {k}:"))
        except ValueError:
            pass
        except KeyError:
            pass

st.markdown("#### 3.b. Scaling")

hs = [0.5, 3.0, 4.0, 6.0, 10.0]
fig = go.Figure()
for h in hs:
    r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for n in options_n]
    fig.add_trace(
        go.Scatter(
            x=options_n,
            y=r,
            name=f"h = {h}",
            mode="lines+markers",
            line={"dash": "dash"},
            marker={"size": 10},
        )
    )
fig.update_layout(
    title="Fig. Finite size scaling of averaged gap ratio",
    xaxis_title="System size n",
    yaxis_title="Averaged gap ratio r",
)
st.write(fig)

fig = go.Figure()
for n in options_n:
    r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for h in hs]
    fig.add_trace(
        go.Scatter(
            x=hs,
            y=r,
            name=f"n = {n}",
            mode="lines+markers",
            line={"dash": "dash"},
            marker={"size": 10},
        )
    )
fig.update_layout(
    title="Fig. Finite size scaling of averaged gap ratio",
    xaxis_title="Disorder strength h",
    yaxis_title="Averaged gap ratio r",
)
st.write(fig)

yes_button = st.select_slider(
    "Start computing (the following 2 plots are relatively time-consuming)",
    options=[True, False],
)
if table == "tsdrg" and yes_button:
    fig = go.Figure()
    for chi in options_chi:
        r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for n in options_n]
        fig.add_trace(
            go.Scatter(
                x=options_n,
                y=r,
                name=f"chi = {chi}",
                mode="lines+markers",
                line={"dash": "dash"},
                marker={"size": 10},
            )
        )
    fig.update_layout(
        title="Fig. Finite size scaling of averaged gap ratio",
        xaxis_title="System size n",
        yaxis_title="Averaged gap ratio r",
    )
    st.write(fig)

    fig = go.Figure()
    for chi in options_chi:
        r = [fetch_gap_ratio(n, h, chi=chi, total_sz=total_sz) for h in hs]
        fig.add_trace(
            go.Scatter(
                x=hs,
                y=r,
                name=f"chi = {chi}",
                mode="lines+markers",
                line={"dash": "dash"},
                marker={"size": 10},
            )
        )
    fig.update_layout(
        title="Fig. Finite size scaling of averaged gap ratio",
        xaxis_title="Disorder strength h",
        yaxis_title="Averaged gap ratio r",
    )
    st.write(fig)

