import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft, rfft, rfftfreq, irfft
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
import soundfile as sf

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ upload Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#


def to_librosa(file_uploaded):
    """
        Function to upload file from librosa 

        Parameters
        ----------
        file uploaded 

        Return
        ----------
        y : samples
        sr : sampling rate      
    """
    if file_uploaded is not None:
        samples, sr = librosa.load(file_uploaded)
        return samples, sr
    
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Generate Sliders Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def generate_slider(arr_names, arr_values , n=1):
    """
        Function to generate slider 

        Parameters
        ----------
        arr_names  : label of each slider 
        arr_values : Values to be controlled for each slider
        n          : step of increment or decrement for slider

        Return
        ----------
        slider_values     
    """
    slider_values = []
    # number of columns for styling = number of label
    col = st.columns(len(arr_names))
    for i in range(len(arr_names)):
        with col[i]:
            tuple = arr_values[i]
            slider = svs.vertical_slider(
                key=arr_names[i], min_value=tuple[0], max_value=tuple[1], default_value=tuple[2], step=n)
            slider_values.append(slider)
            st.write(arr_names[i])

    return slider_values

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Fourier Transform Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def fourier_transform(audio_file, sample_rate):
    """
        Function to Apply fourier transform

        Parameters
        ----------
        audio_file  : file uploaded
        sample_rate : sample rate resulted from librosa

        Return
        ----------
        magnitude 
        frequency     
    """
    number_of_samples = len(audio_file)
    T = 1 / sample_rate # period
    magnitude = rfft(audio_file)
    frequency = rfftfreq(number_of_samples, T)
    return magnitude, frequency


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Modification of signals Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def modifiy_general_signal(magnitude_freq, sliders_value):
    """
        Function to apply changes in frequency 

        Parameters
        ----------
        magnitude_freq : magnitude in frequency domain which you want to change it.
        sliders_value  : value to select the range of frequency to change magnitude.

        Return
        ----------
        magintude_freq : magnitude after apply changes.
    """
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1
    magnitude_band = int(len(magnitude_freq)/10)
    for i in range(len(sliders_value)):
        magnitude_freq[magnitude_band *i:magnitude_band*(i+1)]*= sliders_value[i]
    
    return magnitude_freq


def modifiy_music_signal(magnitude_freq, freqency, sliders_value):
    """
        Function to apply changes in musical instrument 

        Parameters
        ----------
        magnitude_freq : magnitude of the frequency which you want to be changed
        frequency      : frequency to be changed
        sliders_value  : value to be changed on the frequency

        Return
        ----------
        magintude_freq : magnitude after apply changes 
             
    """
     
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1
# looping on the frequency values to multiply the slider values to the frequency of each musical instrument
    counter=0
    for value in freqency :
        if value > 0 and value < 500:
            magnitude_freq[counter] *= sliders_value[0]
        elif value > 500 and value < 1000:
            magnitude_freq[counter] *= sliders_value[1]
        elif value > 1000 and value < 2000:
            magnitude_freq[counter] *= sliders_value[2]
        elif value > 2000 and value < 5000:
            magnitude_freq[counter] *= sliders_value[3]
        counter+=1
    return magnitude_freq



def modifiy_medical_signal(Ecg_file, sliders_value):
    """
        Function to apply changes in medical instrument 

        Parameters
        ----------
        Ecg_file       : CSV file of ECG 
        sliders_value  : value to be changed on the frequency

        Return
        ----------
        magintude_freq : magnitude after apply changes 
             
    """
    fig1 = go.Figure()
    # set x axis label
    fig1.update_xaxes(
        title_text="frequency",  # label
        title_font={"size": 20},
        title_standoff=25)
    # set y axis label
    fig1.update_yaxes(
        title_text="Amplitude(mv)",
        title_font={"size": 20},
    # label
    title_standoff=25)
    
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1
    
    time = Ecg_file.iloc[:, 0]
    magnitude = Ecg_file.iloc[:, 1]
    sample_period = time[1]-time[0]
    n_samples = len(time)
    fourier = rfft(magnitude)
    frequencies = rfftfreq(n_samples, sample_period)
    counter = 0
    for value in frequencies:
        if value > 130:
            fourier[counter] *= (sliders_value[0])
        if value < 130 and value > 80:
            fourier[counter] *= (sliders_value[1])
        if value < 80:
            fourier[counter] *= (sliders_value[2])
        counter += 1
        time_domain_amplitude = np.real(irfft(fourier))
    
    fig_sig = fig1.add_scatter(x=time, y=time_domain_amplitude)
    st.plotly_chart(fig_sig, use_container_width=True)
    return time_domain_amplitude


def modifiy_Pitch_signal(sample, sample_rate, sliders_value):
    """
        Function to apply changes in Pitch
        just shift the signal not change in magintude 

        Parameters
        ----------
        sample         : sample from librosa
        sample_rate    : sample rate from librosa
        sliders_value  : value to be changed on the frequency

        Return
        ----------
        magintude_freq : magnitude after apply changes 
             
    """
   
    # for i in range(len(sliders_value)):
    if sliders_value == None:
        sliders_value = 5
    Pitched_amplitude = librosa.effects.pitch_shift(
        y=sample, sr=sample_rate, n_steps=sliders_value)
    return Pitched_amplitude

def modifiy_vowels_signal(magnitude_freq, freqency, sliders_value):
    """
        Function to apply changes in vowels 

        Parameters
        ----------
        Ecg_file       : CSV file of ECG 
        sliders_value  : value to be changed on the frequency

        Return
        ----------
        magintude_freq : magnitude after apply changes 
             
    """
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1
# looping on the frequency values to multiply the slider values to the frequency of each musical instrument
    counter = 0
    for value in freqency:
        if value > 1900 and value < 5000:
            magnitude_freq[counter] *= sliders_value[0]
        if value > 1500 and value < 3000:
            magnitude_freq[counter] *= sliders_value[1]
        if value > 500 and value < 2000:
            magnitude_freq[counter] *= sliders_value[2]
        range1 = value > 100 and value < 1400
        range2 = value > 2000 and value < 6000
        if range1 or range2:
            magnitude_freq[counter] *= sliders_value[4]
        if value > 490 and value < 2800:
            magnitude_freq[counter] *= sliders_value[3]
        counter += 1
    return magnitude_freq
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Inverse Fourier Transform Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def Inverse_fourier_transform(magnitude_freq_domain):
    """
        Function to apply inverse fourier to turn to convertthe signal into time domain

        Parameters
        ----------
        magnitude_freq_domain

        Return
        ----------
        real part of the magintude in time domain
             
    """
    magnitude_time_domain = np.fft.irfft(magnitude_freq_domain)
    return np.real(magnitude_time_domain)

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Animation Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#
def plot_animation(df):
    """
        Function to make the signal animated

        Parameters
        ----------
        df  : dataframe to be animated

        Return
        ----------
        figure             
    """ 
    # brush          -> for zooming 
    #.interactive()  -> to able zooming
    brush = alt.selection_interval() 
    chart1 = alt.Chart(df).mark_line().encode(
            x=alt.X('time', axis=alt.Axis(title='Time')),
        ).properties(
            width=400,
            height=100
        ).add_selection(
            brush).interactive()
    
    figure = chart1.encode(
                y=alt.Y('amplitude',axis=alt.Axis(title='Amplitude')))| chart1.encode(
                y=alt.Y('amplitude after processing',axis=alt.Axis(title='Amplitude after'))).add_selection(
            brush)
    return figure

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Plot Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def show_plot(samples,samples_after_moidifcation , sampling_rate ):
    """
        Function to show plot

        Parameters
        ----------
        samples                      : samples from librosa
        samples_after_moidifcation   : samples after apply changes
        sampling_rate                : sampling rate from librosa

        Return
        ----------
        None             
    """ 
    time_before = np.array(range(0, len(samples)))/(sampling_rate)
    time_after = np.array(range(0, len(samples)))/(sampling_rate)

    df_afterUpload = pd.DataFrame({'time': time_before[::500], 'amplitude': samples[::500], }, columns=['time',
                                                                                                        'amplitude'])
    df_afterInverse = pd.DataFrame({'time_after': time_after[::500], 'amplitude after processing':
                                    samples_after_moidifcation[::500]}, columns=['time_after', 'amplitude after processing'])
    # merge the 2 dataframes
    common_df=df_afterUpload.merge(df_afterInverse,left_on='time', right_on='time_after')
    # delete time after col to avoid repeatition
    common_df.pop("time_after")

    lines = plot_animation(common_df)
    line_plot = st.altair_chart(lines)
    number_of_element = df_afterInverse.shape[0]  # number of elements in the dataframe
    burst = 10  # number of elements (months) to add to the plot
    size = burst
    for i in range(1, number_of_element):
        # iloc -> comvert dataframes to array
        step_df = common_df.iloc[0:size] 
        lines = plot_animation(step_df)
        line_plot = line_plot.altair_chart(lines)
        size = i + burst
        st.session_state.size = size
        
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _Spectogram Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def spectogram(y,  title_of_graph):
    """
        Function to spectrogram

        Parameters
        ----------
        y
        title_of_graph  

        Return
        ----------
        spectrogram             
    """
    # stft -> have higher resolution than fft to make spectrogram smooth 
    samples_after_fourier = librosa.stft(y)  

    # apply logarithm to cast amplitude to Decibels
    amplitude_db = librosa.amplitude_to_db(np.abs(samples_after_fourier), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(amplitude_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=title_of_graph)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return plt.gcf()

def audio_after_show(magnitude_time_after_inverse, sample_rate):
    """
        Function to display audio after apply changes

        Parameters
        ----------
        magnitude_time_after_inverse : magnitude in time domain after inverse fourier 
        sample rate  

        Return
        ----------
        none            
    """ 
    st.sidebar.write("## Audio after")
    sf.write("output.wav", magnitude_time_after_inverse, sample_rate)
    st.sidebar.audio("output.wav")
