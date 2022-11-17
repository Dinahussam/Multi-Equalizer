import functions
import my_fun
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
import soundfile as sf


########################### Styling #######################################
st.set_page_config(
    page_title="Equalizer",
    page_icon="✅",
    layout="wide",
)

with open("style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)
    st.set_option('deprecation.showPyplotGlobalUse', False)

########################### Sidebar #######################################

with st.sidebar:
    st.write("# Equalizer")
    file_uploaded = st.file_uploader("", accept_multiple_files=False)

    st.write("## Choose The Mode ")

    Mode = st.selectbox(label="", options=[
                            'Frequency', 'Vowels', 'Music instrument', 'Medical', 'Pitch'])


########################### Main #######################################

def main():
    if file_uploaded:
        if file_uploaded.type == "audio/wav" or file_uploaded.type == "audio/mpeg":
            magnitude_at_time, sample_rate = my_fun.to_librosa(file_uploaded)
            spectogram = st.sidebar.checkbox(label="Spectogram")
            st.sidebar.write("## Audio before")

            st.sidebar.audio(file_uploaded.name)
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            if Mode == 'Frequency':

                names = ["0:1000", "1000:2000", "2000:3000", "3000:4000", "4000:5000",
                        "5000:6000", "6000:7000", "7000:8000", "8000:9000", "9000:10000"]
                values_slider = [[0, 10, 1]]*10
                # return slider value
                slider = my_fun.generate_slider(names, values_slider)
            # fourier transform
                magnitude_freq_domain, frequency_freq_domain = my_fun.fourier_transform(
                    magnitude_at_time, sample_rate)
                magnitude_after_modifiy = my_fun.modifiy_general_signal(
                    magnitude_freq_domain, slider)
                magnitude_time_after_inverse = my_fun.Inverse_fourier_transform(
                    magnitude_after_modifiy)
                my_fun.audio_after_show(magnitude_time_after_inverse, sample_rate)
                with col1:
                    # draw the original and modified time domain signal
                    my_fun.show_plot(magnitude_at_time,
                                    magnitude_time_after_inverse, sample_rate)
                if spectogram:
                    # spectogram plotting
                    with col3:
                        st.pyplot(my_fun.spectogram(
                            magnitude_at_time, "Before"))
                    with col4:
                        st.pyplot(my_fun.spectogram(
                            magnitude_time_after_inverse, "After"))

            elif Mode == "Vowels":
                names = ["h", "R", "O", "Y", "L"]
                values_slider = [[0, 10, 1]]*len(names)
                slider = my_fun.generate_slider(names, values_slider)
                magnitude_freq_domain, frequency_freq_domain = my_fun.fourier_transform(
                    magnitude_at_time, sample_rate)
                magnitude_after_modifiy = my_fun.modifiy_vowels_signal(
                    magnitude_freq_domain, frequency_freq_domain, slider)
                magnitude_time_after_inverse = my_fun.Inverse_fourier_transform(
                    magnitude_after_modifiy)
                my_fun.audio_after_show(magnitude_time_after_inverse, sample_rate)

                with col1:
                    # draw the original and modified time domain signal
                    my_fun.show_plot(magnitude_at_time,
                                     magnitude_time_after_inverse, sample_rate)
                if spectogram:
                    # spectogram plotting
                    with col3:
                        st.pyplot(my_fun.spectogram(
                            magnitude_at_time, "Before"))
                    with col4:
                        st.pyplot(my_fun.spectogram(
                            magnitude_time_after_inverse, "After"))

            elif Mode == "Music instrument":
                names = ["Drum ", "Flute", "Key", "Piano"]
                values_slider = [[0, 10, 1]]*len(names)
                slider = my_fun.generate_slider(names, values_slider)
                magnitude_freq_domain, frequency_freq_domain = my_fun.fourier_transform(
                    magnitude_at_time, sample_rate)
                magnitude_after_modifiy = my_fun.modifiy_music_signal(
                    magnitude_freq_domain, frequency_freq_domain, slider)
                magnitude_time_after_inverse = my_fun.Inverse_fourier_transform(
                    magnitude_after_modifiy)
                my_fun.audio_after_show(magnitude_time_after_inverse, sample_rate)

                with col1:
                    my_fun.show_plot(magnitude_at_time,
                                     magnitude_time_after_inverse, sample_rate)

                if spectogram:
                    with col3:
                        st.pyplot(my_fun.spectogram(
                            magnitude_at_time, "Before"))
                    with col4:
                        st.pyplot(my_fun.spectogram(
                            magnitude_time_after_inverse, "After"))

            elif Mode == 'Pitch':
             
                slider = st.slider(label="Pitch Shift",max_value=12, min_value=-12, step=1, value=5)               
                magnitude_after_modifiy = my_fun.modifiy_Pitch_signal(
                magnitude_at_time, sample_rate, slider)
                my_fun.audio_after_show(magnitude_after_modifiy, sample_rate)

                with col1:
                    my_fun.show_plot(magnitude_at_time,
                                     magnitude_after_modifiy, sample_rate)

                if spectogram:
                    with col3:
                        st.pyplot(my_fun.spectogram(
                            magnitude_at_time, "Before"))
                    with col4:
                        st.pyplot(my_fun.spectogram(
                            magnitude_after_modifiy, "After"))

        elif Mode == "Medical":
            names = ["arrythmia 1", "arrythmia 2", "arrythmia 3"]
            values_slider = [[0.0, 2.0, 1]]*len(names)
            Data_frame = pd.read_csv(file_uploaded)
            slider = my_fun.generate_slider(names, values_slider, 0.1)
            magnitude_after_modifiy = my_fun.modifiy_medical_signal(
                Data_frame, slider)


if __name__ == "__main__":
    main()
