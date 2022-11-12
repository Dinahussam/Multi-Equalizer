import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft
import sounddevice as sd
import librosa.display
import librosa
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import streamlit_vertical_slider as svs
import streamlit as st


class Functions:
    # To save history when we refresh the page.
    samples = []  # #
    sampling_rate = 0  # #
    y_axis_freq_domain = []
    copy_y_axis_freq_domain = []
    y_axis_time_domain = []
    y_axis_time_domain_ndarray = []
    final_y_axis_time_domain = []
    n = 0
    T = 0
    upload_store = ""
    factor = 0
    time = []
    time_after = []


# @staticmethod
def upload(file_uploaded):
    if file_uploaded is not None:
        Functions.upload_store = file_uploaded
        Functions.samples, Functions.sampling_rate = librosa.load(Functions.upload_store, duration=20)
        st.write(file_uploaded)
        st.write(type(file_uploaded))
        Functions.time = np.array(range(0, len(Functions.samples))) / Functions.sampling_rate
        Functions.time_after = Functions.time[:round(len(Functions.time) / 2)]


# @staticmethod
def sin():
    x = np.arange(0, 4 * np.pi, 0.1)  # start,stop,step
    y = np.sin(x)
    fig = px.line(x=x, y=y, labels={'x': 't', 'y': 'y'})
    return fig


# @staticmethod
def layout_fig(fig):
    fig.update_layout(
        # auto size=False,
        width=450,
        height=250,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=1
        ),
    )
    return fig


# @staticmethod
def verticalSlider(key, min_value, max_value):
    slider_value = svs.vertical_slider(
        key=key,
        step=1,
        min_value=min_value,
        max_value=max_value,
        default_value=1,
        slider_color='#06283D',
        track_color='lightgray',
        thumb_color='#256D85',
    )
    return slider_value


# @staticmethod
def plot(x_axis, y_axis, name_x_axis, name_y_axis):
    fig = px.line(x=x_axis, y=y_axis, labels={'x': name_x_axis, 'y': name_y_axis})
    #     fig, ax = plt.subplots()
    #     ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    #     plt.grid()
    #     plt.xlabel("Frequency")
    #      plt.ylabel("Amplitude")
    return fig.show()


copy_y_axis_freq_domain = Functions.y_axis_freq_domain.copy()


# @staticmethod
def slider_change(start, end, factor, mode):
    Functions.final_y_axis_time_domain = to_freq_domain(start, end, factor)
    return Functions.final_y_axis_time_domain


# @staticmethod
def to_freq_domain(start, end, slider_factor):  # samples represent the audio

    Functions.factor = slider_factor

    Functions.n = len(Functions.samples)
    Functions.T = 1 / Functions.sampling_rate

    Functions.y_axis_freq_domain = fft(Functions.samples)  # we should work on this
    Functions.y_axis_freq_domain = Functions.y_axis_freq_domain[:round(len(Functions.y_axis_freq_domain) / 2)]
    Functions.x_axis_freq_domain = np.linspace(0, int(1.0 / (2.0 * Functions.T)), int(Functions.n / 2))

    # counter = 0
    # for value in x_axis_freq_domain:
    #     if (value >= start) and (value <= end):
    #         y_axis_freq_domain[counter] *= factor
    #     counter += 1

    Functions.y_axis_freq_domain[start:end] = Functions.y_axis_freq_domain[start:end] * Functions.factor
    # f(x) = y = value = Amplitude

    # st.write(Functions.factor)

    y_axis_t_domain = to_time_domain(Functions.y_axis_freq_domain)
    return y_axis_t_domain


# @staticmethod
def to_time_domain(y_axis_freqDomain):
    Functions.y_axis_time_domain_ndarray = np.array(Functions.y_axis_time_domain)
    Functions.y_axis_time_domain_ndarray = np.fft.ifft(y_axis_freqDomain)
    return Functions.y_axis_time_domain_ndarray.real


# @staticmethod
def clear():
    Functions.copy_y_axis_freq_domain = Functions.y_axis_freq_domain.copy()
