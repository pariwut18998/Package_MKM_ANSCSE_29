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

import numpy as np
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
    """Bar chart of the degree-of-rate-control coefficients returned by drc_fn(T).
    bar_color: a single color for every bar, or {step: color} to color bars
    individually (steps missing from the dict fall back to '#1f77b4")."""
    X = drc_fn(T)
    steps = sorted(X)
    vals = [X[s] for s in steps]
    x = labels if labels is not None else [f'step {s}' for s in steps]
    colors = ([bar_color.get(s, '#1f77b4') for s in steps]
              if isinstance(bar_color, dict) else bar_color)
    fig = go.Figure(go.Bar(x=x, y=vals, marker_color=colors))
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


# step 3 (LH) is split into two independently-tunable barriers, one "as measured
# from CO*" and one "as measured from 2O*" — see build_k_COox for how the two
# combine into one physical rate. DRC/label ordering relies on '3_CO' < '3_O'
# sorting lexicographically between '2' and '4' (verified: '2'<'3_CO'<'3_O'<'4').
_COOX_DRC_LABELS = ['step 1\n(CO ads)', 'step 2\n(O₂ dis-ads)',
                    'step 3\n(LH via CO*)', 'step 3\n(LH via 2O*)', 'step 4\n(CO₂ des)']

# colors for the two LH channels' TS curves; reused on the Ea sliders' labels
_TS_COLOR_CO = '#d62728'   # red
_TS_COLOR_O  = '#9467bd'   # purple

# DRC bar colors: step-3 bars match their TS curve/slider color above, so the
# bar plot visually ties back to which Ea barrier it corresponds to
_COOX_BAR_COLORS = {'1': '#2ca02c', '2': '#2ca02c', '4': '#2ca02c',
                    '3_CO': _TS_COLOR_CO, '3_O': _TS_COLOR_O}

# clarifying note on the network view: step 2 (O2 dissociative adsorption) is
# 1 O2 -> 2 O*, so its net rate is intentionally ~half of steps 1/3_CO/3_O/4
# (all 1:1) -- flagged so it isn't mistaken for a bug
_COOX_NET_NOTES = {2: '1 O₂ → 2 O*, so rate is half the others'}

# the 6 states that carry the real kinetics (must match the caller's G_CO_ox keys)
_COOX_ORDER = ['CO(g)+*', '½O₂(g)+*', 'CO*+½O₂+*', 'CO*+O*', 'CO₂*+*', 'CO₂(g)+2*']


def fe_figure_coox(G, EA, w=0.30, w_gas=0.12, state_font_size=10, y_range=None):
    """CO-oxidation free-energy diagram matching the reaction network exactly:
        CO(g)+*  <-> CO*        (step 1, barrierless)
        O2(g)+2* <-> 2O*        (step 2, barrierless)
        CO* & 2O* <-> CO2*+*    (step 3, TWO independent LH transition states:
                                 one measured from CO* [EA['3_CO']], one from
                                 2O* [EA['3_O']] — see build_k_COox for how both
                                 combine into the one physical rate constant)
        CO2*+*   <-> CO2(g)+*   (step 4, barrierless)
    Two independent starting points (CO(g)+*, O2(g)+2*) feed two independent
    branches that both connect DIRECTLY to CO2*+* via their own TS peak — no
    separate co-adsorbed "CO*+O*" plateau is drawn.

    G  : the 6-state dict driving the kinetics (same as _COOX_ORDER, untouched).
         CO(g)+* and ½O2(g)+* are independent entries (not forced equal) --
         G['½O2(g)+*'] feeds the diagram's O2(g)+2* level directly.
    EA : {'3_CO': barrier J/mol, '3_O': barrier J/mol} — mutated in place by the sliders.
    Diagram-only quantity (never fed back into build_k_COox/the kinetics):
      - 2O*'s level is derived additively (non-interacting adsorbates):
            G[2O*] = G[½O2(g)+*] + (G[CO*+O*] - G[CO*])
    """
    g_cog, g_o2g, g_co, g_merged, g_prod, g_final = (G[s] for s in _COOX_ORDER)
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

    # LH transition states (step 3): CO* and 2O* each have their OWN barrier/peak,
    # colored to match their Ea slider (see make_co_ox_panel), both descending
    # into the same CO2*+* product. No on-chart peak label -- color + hover
    # text identify each curve, the name lives on the slider instead.
    xpk = (xs['COs'] + xs['CO2s']) / 2
    for precursor, ea_key, label, color in [('COs', '3_CO', 'Ea(CO*)', _TS_COLOR_CO),
                                            ('Os', '3_O', 'Ea(2O*)', _TS_COLOR_O)]:
        xa, xb = xs[precursor] + widths[precursor], xs['CO2s'] - widths['CO2s']
        ts = ys[precursor] + EA[ea_key] / 1e3
        fig.add_trace(go.Scatter(
            x=[xa, xpk, xb], y=[ys[precursor], ts, ys['CO2s']], mode='lines',
            line=dict(color=color, width=2, shape='spline'), showlegend=False,
            text=[f'G = {ys[precursor]:.1f} kJ/mol',
                  f'TS ({label})<br>{ea_key} = {EA[ea_key]/1e3:.1f} kJ/mol<br>G_TS = {ts:.1f} kJ/mol',
                  f'G = {ys["CO2s"]:.1f} kJ/mol'],
            hovertemplate='%{text}<extra></extra>',
        ))

    fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(visible=False),
                      yaxis=dict(title='Free energy (kJ/mol)', range=y_range),
                      margin=dict(l=55, r=10, t=10, b=10), font=dict(family='Arial'))
    return fig


def make_co_ox_panel(G, EA, labels, edges, step_rates_fn, drc_fn,
                     ea_min=10, ea_max=200, ea_dstep=5, ea_window=None,
                     T0=600, T_min=300, T_max=1200, T_dstep=50,
                     fe_y_range=None, rate_range=None, show=True):
    """CO-oxidation panel: two Ea sliders — LH barrier measured from CO* and from
    2O* respectively (see fe_figure_coox) — plus a T slider drive the free-energy
    diagram + reaction network + degree-of-rate-control bar chart. Each Ea slider
    carries a colored label (matching its TS curve's color) instead of the plot.
    EA (dict, J/mol, keys '3_CO'/'3_O') is mutated in place; drc_fn(T) -> {step: X}.
    labels/edges : reaction-network layout (see package_test.graph_rnx); edges'
                   steps must include the split '3_CO'/'3_O' channels, not a
                   combined step 3.
    step_rates_fn(T) -> {step: net_rate} : rates for the network view, keeping
                   the '3_CO'/'3_O' channels separate (NOT the averaged r3).
    ea_window : if set (kJ/mol), each slider spans its OWN origin Ea (from EA) +/- ea_window,
                overriding ea_min/ea_max. If None, both sliders share [ea_min, ea_max].
    fe_y_range : optional [ymin, ymax] (kJ/mol) fixing the free-energy diagram's
                 y-axis so it doesn't rescale every time an Ea slider moves.
                 If None, the axis auto-scales.
    rate_range : optional (lo, hi) log10(net rate) fixing the network's color scale.
                 If None (default), it is recomputed from the current rates on every
                 update -- T dominates the rate magnitude far more than Ea, so a
                 fixed range would wash out contrast across the T slider's span."""
    def _bounds(key):
        if ea_window is None:
            return ea_min, ea_max
        origin = int(EA[key] / 1e3)
        return origin - ea_window, origin + ea_window

    ea_co_label = widgets.HTML(f'<b style="color:{_TS_COLOR_CO}">Ea(CO*)</b>')
    lo_co, hi_co = _bounds('3_CO')
    ea_co_slider = widgets.IntSlider(value=int(EA['3_CO'] / 1e3), min=lo_co, max=hi_co,
                                     step=ea_dstep, description='(kJ/mol)',
                                     style={'description_width': '70px'})
    ea_o_label = widgets.HTML(f'<b style="color:{_TS_COLOR_O}">Ea(2O*)</b>')
    lo_o, hi_o = _bounds('3_O')
    ea_o_slider = widgets.IntSlider(value=int(EA['3_O'] / 1e3), min=lo_o, max=hi_o,
                                    step=ea_dstep, description='(kJ/mol)',
                                    style={'description_width': '70px'})
    T_slider = widgets.IntSlider(value=T0, min=T_min, max=T_max, step=T_dstep,
                                 description='T (K)', style={'description_width': '110px'})
    fe_out, net_out, drc_out = widgets.Output(), widgets.Output(), widgets.Output()

    def update(change=None):
        EA['3_CO'] = ea_co_slider.value * 1e3      # mutate caller's EA dict in place
        EA['3_O']  = ea_o_slider.value * 1e3
        T = T_slider.value
        with fe_out:
            clear_output(wait=True)
            display(fe_figure_coox(G, EA, y_range=fe_y_range))
        with net_out:
            clear_output(wait=True)
            net = build_network(labels, edges)
            rates = step_rates_fn(T)
            set_rates(net, rates)
            if rate_range is None:
                logs = np.log10([abs(v) for v in rates.values() if v != 0])
                rr = (float(np.floor(logs.min())) - 1, float(np.ceil(logs.max())) + 1)
            else:
                rr = rate_range
            display(draw_network(net, rate_range=rr, notes=_COOX_NET_NOTES))
        with drc_out:
            clear_output(wait=True)
            display(_drc_figure(drc_fn, T, labels=_COOX_DRC_LABELS, bar_color=_COOX_BAR_COLORS))

    ea_co_slider.observe(update, names='value')
    ea_o_slider.observe(update, names='value')
    T_slider.observe(update, names='value')
    update()
    if show:
        display(widgets.VBox([
            widgets.HBox([widgets.VBox([ea_co_label, ea_co_slider]),
                          widgets.VBox([ea_o_label, ea_o_slider]),
                          T_slider]),
            widgets.HBox([fe_out, net_out]),
            drc_out,
        ]))
    return SimpleNamespace(ea_co_slider=ea_co_slider, ea_o_slider=ea_o_slider,
                           T_slider=T_slider, fe_out=fe_out, net_out=net_out,
                           drc_out=drc_out, update=update)


# --- branching A/B -> ... -> F panel (illustrative): two independent 2-step
# surface paths (A*->C*->E*, B*->D*->E*) converge at E*, then continue E*->F*->F(g).
# colors: A-branch red, B-branch purple (matching the CO-ox convention), merged
# spine (E*->F*) green.
_ABF_COLOR_A = '#d62728'   # red
_ABF_COLOR_B = '#9467bd'   # purple
_ABF_COLOR_SPINE = '#2ca02c'   # green

_ABF_XS = {'Ag': 0.0, 'Bg': 0.0, 'As': 1.0, 'Bs': 1.0, 'Cs': 2.0, 'Ds': 2.0,
          'Es': 3.0, 'Fs': 4.0, 'Fg': 5.0}
_ABF_NAMES = {'Ag': 'A(g) + *', 'As': 'A*', 'Cs': 'C*', 'Bg': 'B(g) + *', 'Bs': 'B*',
             'Ds': 'D*', 'Es': 'E*', 'Fs': 'F*', 'Fg': 'F(g) + *'}
_ABF_TST_EDGES = [(2, 'As', 'Cs', _ABF_COLOR_A), (3, 'Cs', 'Es', _ABF_COLOR_A),
                  (5, 'Bs', 'Ds', _ABF_COLOR_B), (6, 'Ds', 'Es', _ABF_COLOR_B),
                  (7, 'Es', 'Fs', _ABF_COLOR_SPINE)]


def fe_figure_abf(G, EA, w=0.30, w_gas=0.12, state_font_size=10, y_range=None):
    """Branching free-energy diagram: A(g)->A*->C*->E* and B(g)->B*->D*->E* converge
    at E*, then E*->F*->F(g).
    G  : 9-state dict {Ag,As,Cs,Bg,Bs,Ds,Es,Fs,Fg} -> energy (kJ/mol).
    EA : {2,3,5,6,7: barrier J/mol} for the surface TST steps -- mutated in place
         by the sliders."""
    widths = {k: (w_gas if k in ('Ag', 'Bg') else w) for k in _ABF_XS}
    yshifts = {'Ag': 22, 'Bg': -22, 'As': 14, 'Bs': -22, 'Cs': 14, 'Ds': -22,
               'Es': 14, 'Fs': 14, 'Fg': 14}

    fig = go.Figure()
    for key, x in _ABF_XS.items():
        y, bw = G[key], widths[key]
        fig.add_trace(go.Scatter(
            x=[x - bw, x + bw], y=[y, y], mode='lines',
            line=dict(color='black', width=4), showlegend=False,
            text=[f'{_ABF_NAMES[key]}<br>G = {y:.1f} kJ/mol'] * 2,
            hovertemplate='%{text}<extra></extra>',
        ))
        fig.add_annotation(x=x, y=y, yshift=yshifts[key], text=_ABF_NAMES[key],
                           showarrow=False, font=dict(size=state_font_size))

    # barrierless steps: A ads (1), B ads (4), F des (8)
    for a, b in [('Ag', 'As'), ('Bg', 'Bs'), ('Fs', 'Fg')]:
        xa, xb = _ABF_XS[a] + widths[a], _ABF_XS[b] - widths[b]
        fig.add_trace(go.Scatter(x=[xa, xb], y=[G[a], G[b]], mode='lines',
                                 line=dict(color='#1f77b4', width=2, dash='dot'),
                                 showlegend=False, hoverinfo='skip'))

    # TST steps: 2 (As->Cs), 3 (Cs->Es), 5 (Bs->Ds), 6 (Ds->Es), 7 (Es->Fs)
    for step, a, b, color in _ABF_TST_EDGES:
        xa, xb = _ABF_XS[a] + widths[a], _ABF_XS[b] - widths[b]
        xpk = (xa + xb) / 2
        ts = G[a] + EA[step] / 1e3
        fig.add_trace(go.Scatter(
            x=[xa, xpk, xb], y=[G[a], ts, G[b]], mode='lines',
            line=dict(color=color, width=2, shape='spline'), showlegend=False,
            text=[f'G = {G[a]:.1f} kJ/mol',
                  f'TS (step {step})<br>Ea{step} = {EA[step]/1e3:.1f} kJ/mol<br>G_TS = {ts:.1f} kJ/mol',
                  f'G = {G[b]:.1f} kJ/mol'],
            hovertemplate='%{text}<extra></extra>',
        ))
        fig.add_annotation(x=xpk, y=ts, yshift=10, text=f'Ea{step}',
                           showarrow=False, font=dict(color=color, size=11))

    fig.update_layout(height=320, plot_bgcolor='white', paper_bgcolor='white',
                      xaxis=dict(visible=False),
                      yaxis=dict(title='Free energy (kJ/mol)', range=y_range),
                      margin=dict(l=55, r=10, t=10, b=10), font=dict(family='Arial'))
    return fig


def make_abf_panel(G, EA, labels, edges, step_rates_fn, drc_fn,
                   ea_min=10, ea_max=150, ea_window=None, ea_step=5,
                   T0=600, T_min=300, T_max=1200, T_dstep=50,
                   fe_y_range=None, rate_range=None, show=True):
    """Branching A/B -> ... -> F panel: one Ea slider per TST step (2,3,5,6,7) plus
    a T slider drive the free-energy diagram + reaction network + degree-of-rate-
    control bar chart.
    EA (dict, J/mol) is mutated in place; drc_fn(T) -> {step: X}.
    step_rates_fn(T) -> [r_i] (8 net rates, step order 1..8).
    ea_window : if set (kJ/mol), each slider spans its OWN origin Ea (from EA) +/- ea_window,
                overriding ea_min/ea_max. If None, all sliders share [ea_min, ea_max].
    fe_y_range : optional [ymin, ymax] (kJ/mol) fixing the free-energy diagram's y-axis.
    rate_range : optional (lo, hi) log10(net rate) fixing the network's color scale.
                 If None, it is recomputed from the current rates on every update."""
    ea_steps = (2, 3, 5, 6, 7)

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
    T_slider = widgets.IntSlider(value=T0, min=T_min, max=T_max, step=T_dstep,
                                 description='T (K)', style={'description_width': '110px'})
    fe_out, net_out, drc_out = widgets.Output(), widgets.Output(), widgets.Output()

    def update(change=None):
        for i in ea_steps:
            EA[i] = sliders[i].value * 1e3       # mutate caller's EA dict in place
        T = T_slider.value
        with fe_out:
            clear_output(wait=True)
            display(fe_figure_abf(G, EA, y_range=fe_y_range))
        with net_out:
            clear_output(wait=True)
            net = build_network(labels, edges)
            rates = step_rates_fn(T)
            set_rates(net, rates)
            if rate_range is None:
                logs = np.log10([abs(v) for v in rates if v != 0])
                rr = (float(np.floor(logs.min())) - 1, float(np.ceil(logs.max())) + 1)
            else:
                rr = rate_range
            display(draw_network(net, rate_range=rr))
        with drc_out:
            clear_output(wait=True)
            display(_drc_figure(drc_fn, T))

    for s in sliders.values():
        s.observe(update, names='value')
    T_slider.observe(update, names='value')
    update()
    if show:
        display(widgets.VBox([
            widgets.HBox([widgets.Label('Activation Ea:')] + list(sliders.values()) + [T_slider]),
            widgets.HBox([fe_out, net_out]),
            drc_out,
        ]))
    return SimpleNamespace(sliders=sliders, T_slider=T_slider, fe_out=fe_out,
                           net_out=net_out, drc_out=drc_out, update=update)
