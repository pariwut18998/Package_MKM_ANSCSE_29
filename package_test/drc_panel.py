# --- Combined interactive degree-of-rate-control panel for a linear MKM chain ---
# Ea sliders (one per surface TST step) drive three linked views:
#   (1) plotly free-energy diagram, (2) reaction network coloured by net rate,
#   (3) degree-of-rate-control bar chart.
# The physics stays in the caller: pass `step_rates_fn(T) -> [r_i]` and
# `drc_fn(T) -> {step: X}`. The caller's `EA` dict (barriers in J/mol) is mutated
# IN PLACE as the sliders move, so the caller's physics — which reads that same
# dict — sees every change.
#
# Usage in a notebook:
#     from package_test.drc_panel import make_drc_panel
#     make_drc_panel(G_AD, _ORDER_AD, EA_AD, (2, 3, 4), labels_AD, edges_AD,
#                    step_rates_AD, degree_of_rate_control_AD, T_AD)
from types import SimpleNamespace

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
from package_test.graph_rnx import build_network, set_rates, draw_network

_BARW = 0.30


def _fe_figure(G, order, EA, ea_steps, w=_BARW, state_font_size=12, y_range=None):
    """Free-energy diagram. State s spans index i; step (i+1) links state i->i+1:
    steps in `ea_steps` get a TST hump (Ea from `EA`, J/mol), the rest are barrierless."""
    levels = [G[s] for s in order]
    fig = go.Figure()
    for i, g in enumerate(levels):
        # hoverable energy level: shows state name and G value
        fig.add_trace(go.Scatter(
            x=[i - w, i + w], y=[g, g], mode='lines',
            line=dict(color='black', width=4),
            showlegend=False,
            text=[f'{order[i]}<br>G = {g:.1f} kJ/mol'] * 2,
            hovertemplate='%{text}<extra></extra>',
        ))
        fig.add_annotation(x=i, y=g, yshift=14, text=order[i],
                           showarrow=False, font=dict(size=state_font_size))
    for i in range(len(levels) - 1):                   # ads / des: barrierless
        if (i + 1) in ea_steps:
            continue
        a, b = i, i + 1
        fig.add_trace(go.Scatter(x=[a + w, b - w], y=[levels[a], levels[b]], mode='lines',
                                 line=dict(color='#1f77b4', width=2, dash='dot'),
                                 showlegend=False, hoverinfo='skip'))
    for step in sorted(ea_steps):                      # tst barriers
        a, b = step - 1, step
        ts = levels[a] + EA[step] / 1e3
        xpk = (a + b) / 2
        # hoverable TS peak: shows Ea and TS energy
        fig.add_trace(go.Scatter(
            x=[a + w, xpk, b - w], y=[levels[a], ts, levels[b]], mode='lines',
            line=dict(color='#d62728', width=2, shape='spline'),
            showlegend=False,
            text=[f'G = {levels[a]:.1f} kJ/mol',
                  f'TS (step {step})<br>Ea{step} = {EA[step]/1e3:.1f} kJ/mol<br>G_TS = {ts:.1f} kJ/mol',
                  f'G = {levels[b]:.1f} kJ/mol'],
            hovertemplate='%{text}<extra></extra>',
        ))
        fig.add_annotation(x=xpk, y=ts, yshift=10, text=f'Ea{step}',
                           showarrow=False, font=dict(color='#d62728', size=11))
    fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(visible=False),
                      yaxis=dict(title='Free energy (kJ/mol)', range=y_range),
                      margin=dict(l=55, r=10, t=10, b=10), font=dict(family='Arial'))
    return fig


def _drc_figure(drc_fn, T, labels=None, bar_color='#1f77b4'):
    """Bar chart of the degree-of-rate-control coefficients returned by drc_fn(T)."""
    X = drc_fn(T)
    steps = sorted(X)
    vals = [X[s] for s in steps]
    x = labels if labels is not None else [f'step {s}' for s in steps]
    fig = go.Figure(go.Bar(x=x, y=vals, marker_color=bar_color))
    fig.update_layout(height=300, plot_bgcolor='white',
                      yaxis=dict(title='X_RC', range=[-0.2, 1.1], zeroline=True,
                                 zerolinecolor='#bbbbbb'),
                      title=f'Degree of Rate Control @ {T} K  (sum = {sum(vals):.2f})',
                      margin=dict(l=55, r=10, t=40, b=30), font=dict(family='Arial'))
    return fig


def make_drc_panel(G, order, EA, ea_steps, labels, edges,
                   step_rates_fn, drc_fn, T,
                   ea_min=10, ea_max=150, ea_window=None, ea_step=5,
                   fe_y_range=None, show=True):
    """Build the Ea-slider panel driving fe diagram + network + DRC bar.

    G     : {state: energy kJ/mol}      order : state names in diagram order
    EA    : {step: barrier J/mol}  (mutated in place by the sliders)
    ea_steps : surface steps that carry an Ea slider
    labels/edges : reaction-network layout (see package_test.graph_rnx)
    ea_window : if set (kJ/mol), each slider spans its OWN origin Ea (from EA) +/- ea_window,
                overriding ea_min/ea_max. If None, all sliders share [ea_min, ea_max].
    fe_y_range : optional [ymin, ymax] (kJ/mol) fixing the free-energy diagram's y-axis.
                 If None, the axis auto-scales.
    step_rates_fn(T) -> [r_i]   drc_fn(T) -> {step: X}   : caller-side physics
    Returns a handle with `sliders`, `fe_out`, `net_out`, `drc_out`, `update`."""
    ea_steps = tuple(ea_steps)

    def _bounds(i):
        if ea_window is None:
            return ea_min, ea_max
        origin = EA[i] / 1e3
        return origin - ea_window, origin + ea_window

    sliders = {}
    for i in ea_steps:
        lo, hi = _bounds(i)
        sliders[i] = widgets.FloatSlider(value=EA[i] / 1e3, min=lo, max=hi, step=ea_step,
                                         description=f'Ea{i} (kJ/mol)', readout_format='.0f',
                                         continuous_update=False)
    fe_out, net_out, drc_out = widgets.Output(), widgets.Output(), widgets.Output()

    def update(change=None):
        for i in ea_steps:
            EA[i] = sliders[i].value * 1e3       # mutate caller's EA dict in place
        net = build_network(labels, edges)
        set_rates(net, step_rates_fn(T))
        with fe_out:
            clear_output(wait=True); display(_fe_figure(G, order, EA, ea_steps, y_range=fe_y_range))
        with net_out:
            clear_output(wait=True); display(draw_network(net))
        with drc_out:
            clear_output(wait=True); display(_drc_figure(drc_fn, T))

    for s in sliders.values():
        s.observe(update, names='value')
    update()
    if show:
        display(widgets.VBox([
            widgets.HBox([widgets.Label('Activation Ea:')] + list(sliders.values())),
            widgets.HBox([fe_out, net_out]),
            drc_out,
        ]))
    return SimpleNamespace(sliders=sliders, fe_out=fe_out, net_out=net_out,
                           drc_out=drc_out, update=update)


_COOX_DRC_LABELS = ['step 1\n(CO ads)', 'step 2\n(O₂ dis-ads)',
                    'step 3\n(LH rxn)', 'step 4\n(CO₂ des)']

# the 5 states that carry the real kinetics (must match the caller's G_CO_ox keys)
_COOX_ORDER = ['CO(g)+½O₂+2*', 'CO*+½O₂+*', 'CO*+O*', 'CO₂*+*', 'CO₂(g)+2*']


def _fe_figure_coox(G, EA, ea_step=3, w=0.30, w_gas=0.12, state_font_size=10, y_range=None):
    """CO-oxidation free-energy diagram matching the reaction network exactly:
        CO(g)+*  <-> CO*        (step 1, barrierless)
        O2(g)+2* <-> 2O*        (step 2, barrierless)
        CO* & 2O* <-> CO2*+*    (step 3, LH transition state)
        CO2*+*   <-> CO2(g)+*   (step 4, barrierless)
    Two independent starting points (CO(g)+*, O2(g)+2*) feed two independent
    branches that both connect DIRECTLY to CO2*+* via the LH step — no
    separate co-adsorbed "CO*+O*" plateau is drawn.

    G : the 5-state dict driving the kinetics (same as _COOX_ORDER, untouched).
    Diagram-only quantities (never fed back into build_k_COox/the kinetics):
      - O2(g)+2* is drawn at the same reference height as CO(g)+* (both start
        "unreacted").
      - 2O*'s level is derived additively (non-interacting adsorbates):
            G[2O*] = G[O2(g)+2*] + (G[CO*+O*] - G[CO*])
      - the TS peak (step 3) is still measured from the real co-adsorbed
        CO*+O* energy in G, so the barrier height stays kinetically accurate
        even though that state isn't drawn as its own level.
    """
    g_cog, g_co, g_merged, g_prod, g_final = (G[s] for s in _COOX_ORDER)
    g_o2g = g_cog                                  # both branches start "unreacted"
    g_o = g_o2g + (g_merged - g_co)                # additive 2O* branch level

    xs = {'COg': 0.0, 'O2g': 0.0, 'COs': 1.0, 'Os': 1.0, 'CO2s': 2.0, 'CO2g': 3.0}
    ys = {'COg': g_cog, 'O2g': g_o2g, 'COs': g_co, 'Os': g_o, 'CO2s': g_prod, 'CO2g': g_final}
    labels = {'COg': 'CO(g) + *', 'O2g': 'O₂(g) + 2*', 'COs': 'CO*', 'Os': '2O*',
             'CO2s': 'CO₂* + *', 'CO2g': 'CO₂(g) + *'}
    widths = {'COg': w_gas, 'O2g': w_gas, 'COs': w, 'Os': w, 'CO2s': w, 'CO2g': w}
    yshifts = {'COg': 22, 'O2g': -22, 'COs': 14, 'Os': 14, 'CO2s': 14, 'CO2g': 14}

    fig = go.Figure()
    for key, x in xs.items():
        y, bw = ys[key], widths[key]
        # hoverable energy level: shows state name and G value
        fig.add_trace(go.Scatter(
            x=[x - bw, x + bw], y=[y, y], mode='lines',
            line=dict(color='black', width=4), showlegend=False,
            text=[f'{labels[key]}<br>G = {y:.1f} kJ/mol'] * 2,
            hovertemplate='%{text}<extra></extra>',
        ))
        fig.add_annotation(x=x, y=y, yshift=yshifts[key], text=labels[key],
                           showarrow=False, font=dict(size=state_font_size))

    # barrierless steps: CO ads (step 1), O2 dis-ads (step 2), CO2 des (step 4)
    for a, b in [('COg', 'COs'), ('O2g', 'Os'), ('CO2s', 'CO2g')]:
        xa, xb = xs[a] + widths[a], xs[b] - widths[b]
        fig.add_trace(go.Scatter(x=[xa, xb], y=[ys[a], ys[b]], mode='lines',
                                 line=dict(color='#1f77b4', width=2, dash='dot'),
                                 showlegend=False, hoverinfo='skip'))

    # LH transition state (step 3): CO* AND 2O* both feed into it -> CO2*+*.
    # peak height uses the real co-adsorbed CO*+O* energy (g_merged), not drawn as its own level
    ts = g_merged + EA[ea_step] / 1e3
    xpk = (xs['COs'] + xs['CO2s']) / 2
    for precursor in ('COs', 'Os'):
        xa, xb = xs[precursor] + widths[precursor], xs['CO2s'] - widths['CO2s']
        fig.add_trace(go.Scatter(
            x=[xa, xpk, xb], y=[ys[precursor], ts, ys['CO2s']], mode='lines',
            line=dict(color='#d62728', width=2, shape='spline'), showlegend=False,
            text=[f'G = {ys[precursor]:.1f} kJ/mol',
                  f'TS (step {ea_step})<br>Ea{ea_step} = {EA[ea_step]/1e3:.1f} kJ/mol<br>G_TS = {ts:.1f} kJ/mol',
                  f'G = {ys["CO2s"]:.1f} kJ/mol'],
            hovertemplate='%{text}<extra></extra>',
        ))
    fig.add_annotation(x=xpk, y=ts, yshift=10, text=f'Ea{ea_step}',
                       showarrow=False, font=dict(color='#d62728', size=11))

    fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(visible=False),
                      yaxis=dict(title='Free energy (kJ/mol)', range=y_range),
                      margin=dict(l=55, r=10, t=10, b=10), font=dict(family='Arial'))
    return fig


def make_co_ox_panel(G, EA, drc_fn, ea_step=3,
                     ea_min=10, ea_max=200, ea_dstep=5,
                     T0=600, T_min=300, T_max=1200, T_dstep=50, show=True):
    """CO-oxidation panel: an Ea slider (for step `ea_step`) and a T slider drive
    a free-energy diagram (CO*/O* branching, see _fe_figure_coox) + degree-of-
    rate-control bar chart (no network view).
    EA (dict, J/mol) is mutated in place; drc_fn(T) -> {step: X}."""
    ea_slider = widgets.IntSlider(value=int(EA[ea_step] / 1e3), min=ea_min, max=ea_max,
                                  step=ea_dstep, description=f'Ea{ea_step} (kJ/mol)',
                                  style={'description_width': '100px'})
    T_slider = widgets.IntSlider(value=T0, min=T_min, max=T_max, step=T_dstep,
                                 description='T (K)', style={'description_width': '100px'})
    fe_out, drc_out = widgets.Output(), widgets.Output()

    def update(change=None):
        EA[ea_step] = ea_slider.value * 1e3       # mutate caller's EA dict in place
        T = T_slider.value
        with fe_out:
            clear_output(wait=True)
            display(_fe_figure_coox(G, EA, ea_step=ea_step))
        with drc_out:
            clear_output(wait=True)
            display(_drc_figure(drc_fn, T, labels=_COOX_DRC_LABELS, bar_color='#2ca02c'))

    ea_slider.observe(update, names='value')
    T_slider.observe(update, names='value')
    update()
    if show:
        display(widgets.VBox([widgets.HBox([ea_slider, T_slider]),
                              widgets.HBox([fe_out, drc_out])]))
    return SimpleNamespace(ea_slider=ea_slider, T_slider=T_slider,
                           fe_out=fe_out, drc_out=drc_out, update=update)
