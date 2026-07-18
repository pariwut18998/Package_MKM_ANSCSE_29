# --- MKM reaction network: build + visualise (NetworkX + Plotly) ---
# build_network(labels, edges) -> DiGraph; set_rates(G, rates) injects net rates;
# draw_network(G) renders the mechanism with edge colour + width encoding the rate.
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

FONT = 'Arial, Helvetica, sans-serif'
_SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def _pow10(k):
    return f"10{str(int(k)).translate(_SUP)}"


def _sci(x):
    """Compact scientific label, e.g. 4.6x10^6 with a unicode superscript."""
    if not np.isfinite(x) or x == 0:
        return '0'
    e = int(np.floor(np.log10(abs(x))))
    return f"{x / 10.0 ** e:.4f}×{_pow10(e)}"


def build_network(labels, edges):
    """Directed graph of the mechanism from `labels` and `edges`.
    net_rate starts at a placeholder; overwrite real values via set_rates()."""
    G = nx.DiGraph()
    for u, v, step, kind in edges:
        G.add_edge(u, v, step=step, kind=kind, net_rate=1.0)
    nx.set_node_attributes(G, labels, 'label')
    return G


def set_rates(G, rates):
    """Inject real net rates: `rates` is {step: value} or a list in step order."""
    lookup = rates if isinstance(rates, dict) else {i + 1: r for i, r in enumerate(rates)}
    for *_, d in G.edges(data=True):
        d['net_rate'] = lookup[d['step']]


# monotone scale: slow = light grey, fast = black (avoids invisible white edges)
_MONO = [[0.0, '#cccccc'], [1.0, '#000000']]


def draw_network(G, colorscale=_MONO, width_range=(3.0, 13.0), rate_range=(0.0, 15.0), notes=None):
    """Publication figure: edge colour AND width encode log10(net rate); log colourbar.
    Layout is automatic, so this works unchanged for any (acyclic) mechanism.
    notes : optional {step: text} -- small italic label placed under an edge's rate
    label, e.g. to flag a stoichiometric factor already baked into that net rate
    (so it isn't mistaken for a bug)."""
    # left-to-right layers from the DAG; order each layer by the barycentre of its
    # predecessors (Sugiyama heuristic) so parallel branches don't cross.
    pos = {}
    for x, layer in enumerate(nx.topological_generations(G)):
        ordered = list(layer) if x == 0 else sorted(
            layer, reverse=True, key=lambda n: float(np.mean(
                [pos[p][1] for p in G.predecessors(n) if p in pos] or [0.0])))
        m = len(ordered)
        for i, n in enumerate(ordered):
            pos[n] = (float(x), (m - 1) / 2 - i)

    edges = list(G.edges(data=True))
    rates = np.array([abs(d['net_rate']) for *_, d in edges], float)
    rates[rates == 0] = np.nan                               # log needs positive
    logr = np.log10(rates)
    lo, hi = rate_range                                      # fixed log10 scale
    norm = np.clip(np.nan_to_num((logr - lo) / ((hi - lo) or 1.0), nan=0.0), 0.0, 1.0)
    colors = sample_colorscale(colorscale, norm.tolist())
    w0, w1 = width_range
    widths = w0 + norm * (w1 - w0)

    fig = go.Figure()
    bx, by, bt = [], [], []                                  # step-badge positions/text
    for (u, v, d), col, wd in zip(edges, colors, widths):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode='lines',
            line=dict(width=float(wd), color=col),
            hovertemplate=f"step {d['step']} ({d['kind']})<br>"
                          f"net rate = {d['net_rate']:.2e} s⁻¹<extra></extra>",
            showlegend=False))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        bx.append(mx); by.append(my); bt.append(str(d['step']))
        fig.add_annotation(x=mx, y=my, yshift=25, text=_sci(d['net_rate']),
                           showarrow=False, font=dict(family=FONT, size=12, color='#333'))
        if notes and d['step'] in notes:
            fig.add_annotation(x=mx, y=my, yshift=40, text=f"<i>{notes[d['step']]}</i>",
                               showarrow=False, font=dict(family=FONT, size=10, color='#888'))

    # step-number badges at edge midpoints (sized to fit the longest step label,
    # e.g. numeric '1' vs multi-char '3_CO')
    badge_size = max(18, 10 + 4 * max(len(t) for t in bt))
    fig.add_trace(go.Scatter(
        x=bx, y=by, mode='markers+text', text=bt,
        textfont=dict(color='white', size=10, family=FONT),
        marker=dict(size=badge_size, color='#37474f', line=dict(color='white', width=1.5)),
        hoverinfo='skip', showlegend=False))

    # nodes (gas = white, surface = light grey) + labels below
    nodes = list(pos)
    node_x = [pos[n][0] for n in nodes]
    node_y = [pos[n][1] for n in nodes]
    node_c = ['#ffffff' if '(g)' in G.nodes[n]['label'] else '#eef1f5' for n in nodes]
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        text=[G.nodes[n]['label'] for n in nodes], textposition='bottom center',
        textfont=dict(family=FONT, size=12, color='black'),
        marker=dict(size=30, color=node_c, line=dict(color='#3a3a3a', width=1.6)),
        hoverinfo='skip', showlegend=False))

    # log colourbar (invisible marker carrying the scale)
    ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(colorscale=colorscale, cmin=lo, cmax=hi, color=[lo], showscale=True,
                    colorbar=dict(title=dict(text='Net rate (s⁻¹)', side='right'),
                                  tickvals=ticks, ticktext=[_pow10(t) for t in ticks],
                                  len=0.8, thickness=14, outlinewidth=1, ticks='outside')),
        hoverinfo='skip', showlegend=False))

    fig.update_layout(
        font=dict(family=FONT), autosize=True, height=320,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x'),
        margin=dict(l=10, r=10, t=10, b=10))
    return fig
