# --- Interactive free-energy diagram for the branching A/B -> F mechanism ---
# A-channel (steps 2,3) and B-channel (steps 5,6) meet at E*, then the shared spine
# E*->F*->F(g). Every state has its own energy slider (gas+site *+A/*+B/*+F and the
# surface intermediates A*..F*); each surface step also keeps an Ea slider (barrier
# above its reactant state). All energies in kJ/mol. Because E* is one shared slider,
# both channels merge there automatically. The implied ΔG of a step (= E_product -
# E_reactant) and the desorption energies are written back into the caller's DATA /
# GASES dicts so rate_constants() stays consistent with the diagram.
#
# Usage in a notebook:
#     from package_test.energy_diagram import make_energy_diagram
#     diagram = make_energy_diagram(DATA, GASES)   # builds + displays the widget
#     diagram.step_table()                         # print Ea / ΔG of every step
from types import SimpleNamespace

import numpy as np
import ipywidgets as widgets
import plotly.graph_objects as go
from scipy.constants import N_A
from IPython.display import display, clear_output

SURF   = [2, 3, 5, 6, 7]                   # surface steps carrying an Ea slider
_KJM2J = 1e3 / N_A                          # kJ/mol -> J/molecule (for write-back to DATA)

_XS   = {'Ag': 0, 'As': 1, 'Cs': 2,        # A-channel  | both channels share E*, F*, F(g)
         'Bg': 0, 'Bs': 1, 'Ds': 2,        # B-channel  |
         'Es': 3, 'Fs': 4, 'Fg': 5}        # shared spine
_NAME = {'Ag': '*+A(g)', 'As': 'A*', 'Cs': 'C*', 'Bg': '*+B(g)', 'Bs': 'B*',
         'Ds': 'D*', 'Es': 'E*', 'Fs': 'F*', 'Fg': '*+F(g)'}
_BLU, _ORG, _GRY = '#1f77b4', '#ff7f0e', '#555555'
_TRANS = [('Ag', 'As', None, _BLU), ('As', 'Cs', 2, _BLU), ('Cs', 'Es', 3, _BLU),
          ('Bg', 'Bs', None, _ORG), ('Bs', 'Ds', 5, _ORG), ('Ds', 'Es', 6, _ORG),
          ('Es', 'Fs', 7, _GRY),    ('Fs', 'Fg', None, _GRY)]
_RXN = {step: (u, v) for u, v, step, _ in _TRANS if step is not None}  # step -> (reac, prod)
_STEP_STATES = {1: ('Ag', 'As'), 2: ('As', 'Cs'), 3: ('Cs', 'Es'),   # every step -> (reac, prod)
                4: ('Bg', 'Bs'), 5: ('Bs', 'Ds'), 6: ('Ds', 'Es'),
                7: ('Es', 'Fs'), 8: ('Fs', 'Fg')}
_BARW = 0.30
_STATE_COLS = (['Ag', 'Bg', 'Fg'],                        # col 1: *+gas references
               ['As', 'Bs', 'Cs', 'Ds', 'Es', 'Fs'])      # col 2: surface intermediates A*..F*
_STATE_ORDER = _STATE_COLS[0] + _STATE_COLS[1]            # *+gas, then A*..F*

# ---- default Ea and state energies (kJ/mol), tuned from the per-step table ----
_EA0 = {2: 50, 3: 50, 5: 50, 6: 95, 7: 150}
_E0  = {'Ag': 0, 'Bg': 0, 'As': -120, 'Bs': -260, 'Cs': -160, 'Ds': -370,
        'Es': -470, 'Fs': -450, 'Fg': -450}


def _smooth(x1, y1, x2, y2, n=24):
    """Cosine-eased curve from (x1,y1) to (x2,y2)."""
    t = np.linspace(0, 1, n)
    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * (1 - np.cos(np.pi * t)) / 2


def energy_fig(L, Ea):
    """L = {state: energy kJ/mol}; Ea = {step: barrier kJ/mol above its reactant}."""
    fig = go.Figure()
    for s, x in _XS.items():               # state level bars + name labels
        fig.add_trace(go.Scatter(x=[x - _BARW, x + _BARW], y=[L[s], L[s]], mode='lines',
                                 line=dict(color='black', width=3),
                                 hoverinfo='skip', showlegend=False))
        fig.add_annotation(x=x, y=L[s], yshift=-15, text=_NAME[s], showarrow=False,
                           font=dict(size=11))
    for u, v, step, col in _TRANS:         # transitions: hump for tst, ease for ads/des
        xs, xe, ys, ye = _XS[u] + _BARW, _XS[v] - _BARW, L[u], L[v]
        if step is None:
            cx, cy = _smooth(xs, ys, xe, ye)
            dash = 'solid'
        else:
            xm, yts = (xs + xe) / 2, L[u] + Ea[step]
            x1, y1 = _smooth(xs, ys, xm, yts)
            x2, y2 = _smooth(xm, yts, xe, ye)
            cx, cy = np.r_[x1, x2], np.r_[y1, y2]
            dash = 'dash'                              # Ea barrier curve drawn dashed
            fig.add_annotation(x=xm, y=yts, yshift=10, text=f"Ea{step}={Ea[step]:.0f}",
                               showarrow=False, font=dict(size=10, color=col))
        fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines',
                                 line=dict(color=col, width=2.5, dash=dash),
                                 hoverinfo='skip', showlegend=False))
    fig.update_layout(autosize=True, height=440, plot_bgcolor='white',   # responsive width (fills container)
                      title='Free-energy diagram  (A-channel blue · B-channel orange)',
                      xaxis=dict(visible=False),
                      yaxis=dict(title='G (kJ/mol)', zeroline=True, gridcolor='#eee'),
                      margin=dict(l=60, r=20, t=40, b=20))
    return fig


def step_table(L, Ea, DATA):
    """Print Ea and ΔG (kJ/mol) of every step; ΔG = E(product) - E(reactant)."""
    print(f"{'step':<5}{'type':<5}{'Ea (kJ/mol)':>12}{'ΔG (kJ/mol)':>14}")
    for s in range(1, 9):
        u, v = _STEP_STATES[s]
        dG = L[v] - L[u]
        ea = f'{Ea[s]:.0f}' if s in SURF else '—'
        print(f"{s:<5}{DATA[s]['type']:<5}{ea:>12}{dG:>+14.0f}")


def make_energy_diagram(DATA, GASES, ea0=None, e0=None, show=True):
    """Build the interactive diagram. Sliders write Ea/ΔG/Edes back into DATA/GASES.

    Returns a handle with `ea_sliders`, `e_sliders`, `out`, `update()` and
    `step_table()` (the latter prints the current per-step Ea/ΔG)."""
    ea0 = dict(_EA0 if ea0 is None else ea0)
    e0  = dict(_E0 if e0 is None else e0)
    ea_sliders = {s: widgets.FloatSlider(value=ea0[s], min=0, max=200, step=5,
                                         description=f'Ea{s}', readout_format='.0f')
                  for s in SURF}
    e_sliders  = {s: widgets.FloatSlider(value=e0[s], min=-600, max=100, step=10,
                                         description=_NAME[s], readout_format='.0f')
                  for s in _STATE_ORDER}
    out = widgets.Output()

    def _state():
        return {s: e_sliders[s].value for s in _XS}

    def _bar():
        return {s: ea_sliders[s].value for s in SURF}

    def apply_energies(data, gases):
        """Write the current slider-derived Ea_f / d_G (per step) and Edes (per gas)
        into `data` / `gases`, in place. Returns the state energies L and barriers Ea."""
        L, Ea = _state(), _bar()
        for step, (u, v) in _RXN.items():      # ΔG of a step = E(product) - E(reactant)
            data[step]['Ea_f'] = Ea[step] * _KJM2J
            data[step]['d_G']  = (L[v] - L[u]) * _KJM2J
        gases['A']['Edes'] = (L['Ag'] - L['As']) * _KJM2J   # desorption energies from levels
        gases['B']['Edes'] = (L['Bg'] - L['Bs']) * _KJM2J
        gases['F']['Edes'] = (L['Fg'] - L['Fs']) * _KJM2J
        return L, Ea

    def update(change=None):
        L, Ea = apply_energies(DATA, GASES)    # keep the caller's DATA / GASES in sync
        with out:
            clear_output(wait=True)
            display(energy_fig(L, Ea))     # display() renders once (fig.show() can double-render in Output)
            #step_table(L, Ea, DATA)

    for w in list(ea_sliders.values()) + list(e_sliders.values()):
        w.observe(update, names='value')
    update()
    if show:
        state_cols = widgets.HBox([widgets.VBox([e_sliders[s] for s in col])
                                   for col in _STATE_COLS])
        display(widgets.HBox([
            widgets.VBox([widgets.Label('Activation Ea (kJ/mol)')] + [ea_sliders[s] for s in SURF]),
            widgets.VBox([widgets.Label('State energy (kJ/mol)'), state_cols]),
        ]))

    return SimpleNamespace(ea_sliders=ea_sliders, e_sliders=e_sliders, out=out,
                           update=update, apply_energies=apply_energies,
                           step_table=lambda: step_table(_state(), _bar(), DATA))
