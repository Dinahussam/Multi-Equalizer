import functions
import streamlit as st
import plotly.express as px
import pandas as pd
import streamlit_vertical_slider as svs
import wave
import struct
from librosa import display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft
import sounddevice as sd
import librosa.display
import librosa
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
import altair as alt


# general styling and tab name
st.set_page_config(
    page_title="Equalizer",
    page_icon="✅",
    layout="wide",
)

# For any warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# styles from css file
with open(r"C:\Users\Function\Desktop\DSP_Task2-main\DSP_Task2-main\style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)

# Lists
sliders_value = []
signal_mode = ['frequency', 'vowels', 'music', 'medical', 'bitch']

# Dictionaries
frequencies_slider = {
    "0:100": [0, 100],  # [start, end]
    "100:200": [100, 200],
    "200:500": [200, 500],
    "500:1000": [500, 1000],
    "1000:4000": [1000, 4000],
}

vowels_sliders = {
    "L": [200, 400],
    "B": [100, 200],
    "M": [500, 1000],
    "K": [10, 100],
    "O": [1000, 4000],
}

music_sliders = {
    "piano1": [200, 400],
    "piano2": [100, 200],
    "piano3": [500, 1000],
    "piano4": [10, 100],
    "piano5": [1000, 4000],
}

medical_sliders = {
    "arrhythmia1": [200],
    "arrhythmia2": [100],
    "arrhythmia3": [1964],
    "arrhythmia4": [8],
    "arrhythmia5": [50],
}

bitch_sliders = {
    "bitch1": [200],
    "bitch2": [100],
    "bitch3": [1964],
    "bitch4": [8],
    "bitch5": [50],
}


signal = st.radio("", ('Frequency', 'Vowels', 'Music Instrumentation', 'Medical Instrumentation', 'Bitch'))

# sidebar components
with st.sidebar:
    # title
    st.write("# Equalizer")

    file_uploaded = st.file_uploader("", type='wav', accept_multiple_files=False)
    functions.upload(file_uploaded)

    # 5 Sliders
    col = st.columns([0.5, 0.5, 0.5, 0.5, 0.5])
    # slider = 0
    for i in range(0, 5, 1):
        with col[i]:
            slider = functions.verticalSlider(i, 0, 100)
            slider = slider or 0
            sliders_value.append(slider)
            # st.write(type(sliders_value[i]))
            # st.write(sliders_value[i])

    if functions.Functions.upload_store:
        st.write("#### Audio before:")
        st.audio(functions.Functions.upload_store)

# Side bar end
# TypeError: int() argument must be a string, a bytes-like object or a number, not 'NoneType'


col_before, col_after = st.columns([2, 2])

with col_before:
    col_before.write("#### Before")

with col_after:
    col_after.write("#### After")


if signal == 'Frequency':
    mode = signal_mode[0]
    index = 0
    for freq_label in frequencies_slider:
        with col[index]:
            st.write(freq_label)
            index += 1
    if functions.Functions.upload_store:
        # functions.plotShow(functions.Functions.samples,functions.Functions.final_y_axis_time_domain)

        # if functions.Functions.upload_store:,functions.Functions.final_y_axis_time_domain
        # wav_fig = px.line(x=functions.Functions.time, y=functions.Functions.samples)

        # fig_layout_wav = functions.layout_fig(wav_fig)
        # col_before.plotly_chart(fig_layout_wav)

        y_axis_after = functions.slider_change(0, 4000, 0, mode)
        with col_before:
            functions.plotShow_before(functions.Functions.samples)

        with col_after:
            functions.plotShow_after(functions.Functions.final_y_axis_time_domain)


        # col_before, col_after = functions.plotShow(functions.Functions.samples, functions.Functions.final_y_axis_time_domain)

        # y_axis_after = functions.slider_change(frequencies_slider["0:100"][0], frequencies_slider["0:100"][1],
        #                                        sliders_value[0], mode)
        # y_axis_after += functions.slider_change(frequencies_slider["100:200"][0], frequencies_slider["100:200"][1],
        #                                         sliders_value[1], mode)
        # y_axis_after += functions.slider_change(frequencies_slider["200:500"][0], frequencies_slider["200:500"][1],
        #                                         sliders_value[2], mode)
        # y_axis_after += functions.slider_change(frequencies_slider["500:1000"][0], frequencies_slider["500:1000"][1],
        #                                         sliders_value[3], mode)
        # y_axis_after += functions.slider_change(frequencies_slider["1000:4000"][0], frequencies_slider["1000:4000"][1]
        #                                         ,sliders_value[4], mode)

        # y_ax = functions.y_axis_time_domain_ndarray

        after_fig = px.line(x=functions.Functions.time_after, y=y_axis_after)
        fig_layout_after = functions.layout_fig(after_fig)
        # col_after.plotly_chart(fig_layout_after)

        Show_spectrogram = st.checkbox('Show spectrogram', value=False)
        if Show_spectrogram:
            with col_before:
                functions.plot_spectrogram(functions.Functions.samples, functions.Functions.sampling_rate)
            with col_after:
                functions.plot_spectrogram(functions.Functions.final_y_axis_time_domain, (functions.Functions.sampling_rate/2))

if signal == 'Vowels':
    mode = signal_mode[1]

    index = 0
    for vowels_label in vowels_sliders:
        with col[index]:
            st.write(vowels_label)
            index += 1

    if functions.Functions.upload_store:
        wav_fig = px.line(x=functions.Functions.time, y=functions.Functions.samples)
        fig_layout = functions.layout_fig(wav_fig)
        col_before.plotly_chart(fig_layout)
        col_after.plotly_chart(fig_layout)
        Show_spectrogram = st.checkbox('Show spectrogram', value=False)
        if Show_spectrogram:
            col_after.plotly_chart(fig_layout)
            col_before.plotly_chart(fig_layout)
# #########################################################################
if signal == 'Music Instrumentation':
    mode = signal_mode[2]

    index = 0
    for music_label in music_sliders:
        with col[index]:
            st.write(music_label)
            index += 1

    if functions.Functions.upload_store:
        wav_fig = px.line(x=functions.Functions.time, y=functions.Functions.samples)
        fig_layout = functions.layout_fig(wav_fig)
        col_before.plotly_chart(fig_layout)
        col_after.plotly_chart(fig_layout)
        Show_spectrogram = st.checkbox('Show spectrogram', value=False)
        if Show_spectrogram:
            col_after.plotly_chart(fig_layout)
            col_before.plotly_chart(fig_layout)
# ##########################################################################
if signal == 'Medical Instrumentation':
    mode = signal_mode[3]

    index = 0
    for medical_label in medical_sliders:
        with col[index]:
            st.write(medical_label)
            index += 1

    if functions.Functions.upload_store:
        wav_fig = px.line(x=functions.Functions.time, y=functions.Functions.samples)
        fig_layout = functions.layout_fig(wav_fig)
        col_before.plotly_chart(fig_layout)
        col_after.plotly_chart(fig_layout)
        Show_spectrogram = st.checkbox('Show spectrogram', value=False)
        if Show_spectrogram:
            col_after.plotly_chart(fig_layout)
            col_before.plotly_chart(fig_layout)
# ##########################################################################
if signal == 'Bitch':
    mode = signal_mode[4]

    index = 0
    for bitch_label in bitch_sliders:
        with col[index]:
            st.write(bitch_label)
            index += 1

    if functions.Functions.upload_store:
        wav_fig = px.line(x=functions.Functions.time, y=functions.Functions.samples)
        fig_layout = functions.layout_fig(wav_fig)
        col_before.plotly_chart(fig_layout)
        col_after.plotly_chart(fig_layout)
        Show_spectrogram = st.checkbox('Show spectrogram', value=False)
        if Show_spectrogram:
            col_after.plotly_chart(fig_layout)
            col_before.plotly_chart(fig_layout)

with st.sidebar:
    if functions.Functions.upload_store:
        st.write("#### Audio after:")
        norm = np.int16(
            functions.Functions.final_y_axis_time_domain * (32767 / functions.Functions.final_y_axis_time_domain.max()))
        write('Edited_audio.wav', round(functions.Functions.sampling_rate), norm)
        st.audio('Edited_audio.wav', format='audio/wav')
