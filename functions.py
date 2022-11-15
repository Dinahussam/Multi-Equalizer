import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft, rfft, rfftfreq
import sounddevice as sd
import librosa.display
import librosa
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import streamlit_vertical_slider as svs
import streamlit as st
import pandas as pd
import altair as alt
import time


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
        # st.write(file_uploaded)
        # st.write(type(file_uploaded))
        Functions.time = np.array(range(0, len(Functions.samples))) / Functions.sampling_rate
        Functions.time_after = np.array(range(0, len(Functions.samples))) / (Functions.sampling_rate / 2)
        Functions.time_after = Functions.time_after[:round((len(Functions.time_after) - 4) / 2)]


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
    #     plt.ylabel("Amplitude")
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

    Functions.y_axis_freq_domain = rfft(Functions.samples)  # we should work on this
    Functions.y_axis_freq_domain = Functions.y_axis_freq_domain[:(len(Functions.y_axis_freq_domain) // 2)]

    Functions.x_axis_freq_domain = np.fft.rfftfreq(Functions.n, Functions.sampling_rate)
    Functions.x_axis_freq_domain = Functions.x_axis_freq_domain[:len(Functions.x_axis_freq_domain) // 2]

    # counter = 0
    # for value in x_axis_freq_domain:
    #     if (value >= start) and (value <= end):
    #         y_axis_freq_domain[counter] *= factor
    #     counter += 1

    Functions.y_axis_freq_domain[start:end] *= Functions.factor  # f(x) = y = value = Amplitude

    # st.write(Functions.factor)

    y_axis_t_domain = to_time_domain(Functions.y_axis_freq_domain)
    return y_axis_t_domain


# @staticmethod
def to_time_domain(y_axis_freqDomain):
    Functions.y_axis_time_domain_ndarray = np.array(Functions.y_axis_time_domain)
    Functions.y_axis_time_domain_ndarray = np.fft.irfft(y_axis_freqDomain)/2
    return Functions.y_axis_time_domain_ndarray.real


# @staticmethod
def clear():
    Functions.copy_y_axis_freq_domain = Functions.y_axis_freq_domain.copy()

def plot_animation(df):
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
        ).properties(
            width=400,
            height=200
        ).add_selection(
            brush).interactive()
    
    figure = chart1.encode(
                  y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                  y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
            brush)
    return figure

def show_plot(data,inverse_data):
    df_afterUpload = pd.DataFrame({'time': Functions.time[::500], 'amplitude': data[::500], }, columns=['time',
                                                                                                        'amplitude'])
    df_afterInverse = pd.DataFrame({'time_after': Functions.time_after[::250], 'amplitude after processing':
                                    inverse_data[::250]}, columns=['time_after', 'amplitude after processing'])
    common_df=df_afterUpload.merge(df_afterInverse,left_on='time', right_on='time_after')
    common_df.pop("time_after")

    lines = plot_animation(common_df)
    line_plot = st.altair_chart(lines)
    N = df_afterInverse.shape[0]  # number of elements in the dataframe
    burst = 10  # number of elements (months) to add to the plot
    size = burst
    for i in range(1, N):
        step_df = common_df.iloc[0:size]
        lines = plot_animation(step_df)
        line_plot = line_plot.altair_chart(lines)
        size = i + burst
        st.session_state.size = size



# Plot Spectrogram
def plot_spectrogram(signal, sample_rate, y_axis="linear"):
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_scale = librosa.stft(signal, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Y_scale = np.abs(S_scale) ** 2
    Y_log_scale = librosa.power_to_db(Y_scale)
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(Y_log_scale, sr=sample_rate, hop_length=512, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(clear_figure=False)


# def plotShow(data, idata):
#     # st.write(len(Functions.time_after))
#     # st.write(len(Functions.final_y_axis_time_domain))
#     time_afterUpload = np.linspace(0, 2, len(data))
#     df_afterUpload = pd.DataFrame({'time': Functions.time[::500], 'amplitude': data[::500], }, columns=['time',
#                                                                                                         'amplitude'])
#     # time_afterInversw = np.linspace(0,2,len(data))
#
#     lines_afterUpload = plot_animation_beforeProcessing(df_afterUpload)
#
#     line_plot1 = st.altair_chart(lines_afterUpload)
#     N1 = df_afterUpload.shape[0]  # number of elements in the dataframe
#     burst1 = 10      # number of elements (months) to add to the plot
#     size1 = burst1    # size of the current dataset
#
#     for i in range(1, N1):
#         # st.session_state.start=i
#         # print(st.session_state.start)
#         step_df1 = df_afterUpload.iloc[0:size1]
#         lines_afterUpload = plot_animation_beforeProcessing(step_df1)
#         line_plot1 = line_plot1.altair_chart(lines_afterUpload)
#         size1 = i + burst1
#         st.session_state.size1 = size1
#   # ##################################################################################################################
#     # lines_afterInverse = plot_animation(df_afterUpload)
#     df_afterInverse = pd.DataFrame({'time': Functions.time_after[::250], 'amplitude after processing': idata[::250]},
#                                    columns=['time', 'amplitude after processing'])
#     lines_afterInverse = plot_animation_afterProcessing(df_afterInverse)
#     line_plot2 = st.altair_chart(lines_afterInverse)
#     N2 = df_afterInverse.shape[0]  # number of elements in the dataframe
#     burst2 = 10      # number of elements (months) to add to the plot
#     size2 = burst2
#     for i in range(1, N2):
#         # st.session_state.start=i
#         # print(st.session_state.start)
#         step_df2 = df_afterInverse.iloc[0:size2]
#         lines_afterInverse = plot_animation_afterProcessing(step_df2)
#         line_plot2 = line_plot2.altair_chart(lines_afterInverse)
#         size2 = i + burst2
#         st.session_state.size2 = size2
#     time.sleep(.1)

# ###################################
    # elif resume_btn: 
    #         print(st.session_state.start)
    #         for i in range( st.session_state.start,N):
    #             st.session_state.start =i 
    #             step_df = df.iloc[0:size]
    #             lines = plot_animation(step_df)
    #             line_plot = line_plot.altair_chart(lines)
    #             st.session_state.size1 = size
    #             size = i + burst
    #             time.sleep(.1)
                 
    #             # if st.session_state.size1 >=N:
    #             #     size = N - 1

    # elif pause_btn:
    #         step_df = df.iloc[0:st.session_state.size1]
    #         lines = plot_animation(step_df)
    #         line_plot = line_plot.altair_chart(lines)
    #         # size = i + burst
    #         if pause_btn:
    #             print("pause")
