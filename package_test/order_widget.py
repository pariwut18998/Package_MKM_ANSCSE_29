# --- Interactive steady-state step-rate bar chart for the A <-> B unimolecular MKM ---
# Two pressure sliders (P_A, P_B) drive a plotly bar chart of the net step rates
# r1, r2, r3 at steady state. The physics stays in the caller: pass a `rate_fn(PA, PB)`
# that returns [r1, r2, r3]. Keeping the solver in the notebook is deliberate — the
# notebook's dydt reads the module-level globals P_A/P_B, so the pressure update must
# happen in the notebook's namespace (see steady_state_rates there).
#
# Usage in a notebook:
#     from package_test.order_widget import make_order_widget
#     make_order_widget(lambda PA, PB: steady_state_rates(1200, PA, PB))
from types import SimpleNamespace

import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display

STEP_LABELS = ['r1: A(g)<->A*', 'r2: A*<->B*', 'r3: B*<->B(g)']
_BAR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def make_order_widget(rate_fn, T=1200, p_min=1e2, p_max=3e3, p_step=1e2,
                      pa0=2e2, pb0=1e2, show=True):
    """Build the interactive steady-state step-rate bar chart.

    rate_fn(PA, PB) -> [r1, r2, r3] : net step rates at steady state.
    Returns a handle with `pa_slider`, `pb_slider`, `fig`, `update`."""
    pa_slider = widgets.FloatSlider(value=pa0, min=p_min, max=p_max, step=p_step,
                                    description='P_A (Pa)', readout_format='.1e')
    pb_slider = widgets.FloatSlider(value=pb0, min=p_min, max=p_max, step=p_step,
                                    description='P_B (Pa)', readout_format='.1e')

    fig = go.FigureWidget(
        data=[go.Bar(x=STEP_LABELS, y=[0, 0, 0], marker_color=_BAR_COLORS)])
    fig.update_layout(title=f'Steady-state step rates @ {T} K',
                      yaxis_title='net rate (1/s)',
                      yaxis_type='log', yaxis_range=[1, 10],
                      width=480, height=380, margin=dict(t=40, b=40))

    def update(change=None):
        with fig.batch_update():
            fig.data[0].y = rate_fn(pa_slider.value, pb_slider.value)

    pa_slider.observe(update, names='value')
    pb_slider.observe(update, names='value')
    update()
    if show:
        display(widgets.HBox([widgets.VBox([pa_slider, pb_slider]), fig]))
    return SimpleNamespace(pa_slider=pa_slider, pb_slider=pb_slider,
                           fig=fig, update=update)
