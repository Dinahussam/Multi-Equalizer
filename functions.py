import numpy as np
import plotly.express as px


class Functions:
    @staticmethod
    def sin():
        x = np.arange(0, 4 * np.pi, 0.1)  # start,stop,step
        y = np.sin(x)
        fig = px.line(x=x, y=y, labels={'x': 't', 'y': 'y'})
        return fig

    @staticmethod
    def layout_fig(fig):
        fig.update_layout(
            # auto size=False,
            width=1000,
            height=300,
            margin=dict(
                l=50,
                r=50,
                b=50,
                t=50,
                pad=1
            ),
        )
        return fig