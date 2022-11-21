import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq, irfft
import librosa.display
import librosa
import plotly.graph_objects as go
import streamlit_vertical_slider as svs
import streamlit as st
import pandas as pd
import altair as alt
import soundfile as sf
import time

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
        y, sr = librosa.load(file_uploaded)
        return y, sr

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Generate Sliders Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#


def generate_slider(arr_names, arr_values, n=1):
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
    # with st.sidebar :
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
    number_samples = len(audio_file)
    T = 1 / sample_rate
    magnitude = rfft(audio_file)
    frequency = rfftfreq(number_samples, T)

    return magnitude, frequency


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Modification of signals Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def modifiy_general_signal(name, magnitude_freq, frequency_freq_domain, sliders_value, ranges):
    """
        Function to apply changes in frequency / musical instrumntation  / vowels

        Parameters
        ----------
        name                  : mode -> freq / music or vowels
        magnitude_freq        : magnitude in frequency domain which you want to change it.
        frequency_freq_domain : frequency after apply fourier transform
        sliders_value         : value to select the range of frequency to change magnitude.
        ranges                : ranges of sliders

        Return
        ----------
        magintude_freq : magnitude after apply changes.
    """
    for i in range(len(sliders_value)):
        if sliders_value[i] == None:
            sliders_value[i] = 1

    for i in range(len(sliders_value)):
        counter = 0
        for value in frequency_freq_domain:
            if value > ranges[name[i]][0] and value < ranges[name[i]][1]:
                magnitude_freq[counter] *= sliders_value[i]
            counter += 1
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
    if sliders_value == None:
        sliders_value = 5
    Pitched_amplitude = librosa.effects.pitch_shift(
        y=sample, sr=sample_rate, n_steps=sliders_value)
    return Pitched_amplitude


def processing(mode, names, values_slider, magnitude_at_time, sample_rate, show_spec, ranges, pitch_step):
    """
        Function to do processing and show spectrogram

        Parameters
        ----------
        mode                  : mode -> freq / music / medical or vowels
        names                 : label of sliders
        magnitude_at_time     : magnitude at time domain
        sample_rate           : sample rate after upload by librosa
        show_spec             : session state of spectrogram
        sliders_value         : value to select the range of frequency to change magnitude.
        ranges                : ranges of sliders
        
        Return
        ----------
        magintude_freq : magnitude after apply changes.
    """

    if mode == 'Frequency' or 'Vowels' or 'Music Instrument':
        # 4 cols for 4 sliders
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)
        slider = generate_slider(names, values_slider)

        magnitude_freq_domain, frequency_freq_domain = fourier_transform(
            magnitude_at_time, sample_rate)

        magnitude_after_modifiy = modifiy_general_signal(
            names, magnitude_freq_domain, frequency_freq_domain, slider, ranges)
            
        magnitude_time_after_inverse = Inverse_fourier_transform(
            magnitude_after_modifiy)

    elif mode == 'Pitch Shift':
        magnitude_time_after_inverse = modifiy_Pitch_signal(
            magnitude_at_time, sample_rate, pitch_step)

    audio_after_show(magnitude_time_after_inverse, sample_rate)
    with col1:
        # draw the original and modified time domain signal
        show_plot(magnitude_at_time,
                  magnitude_time_after_inverse, sample_rate)
    if show_spec == 1:
        # spectogram plotting
        with col3:
            st.pyplot(spectogram(
                magnitude_at_time, "Before"))
        with col4:
            st.pyplot(spectogram(
                magnitude_time_after_inverse, "After"))


#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Audio show Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#


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
    brush = alt.selection_interval()
    chart1 = alt.Chart(df).mark_line().encode(
        x=alt.X('time', axis=alt.Axis(title='Time')),
    ).properties(
        width=400,
        height=200
    ).add_selection(
        brush).interactive()

    figure = chart1.encode(
        y=alt.Y('amplitude', axis=alt.Axis(title='Amplitude'))) | chart1.encode(
        y=alt.Y('amplitude after processing', axis=alt.Axis(title='Amplitude after'))).add_selection(
        brush)
    return figure

#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Plot Functions_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

def currentState(df, size, num_of_element):
        if st.session_state.size1 == 0:
            step_df = df.iloc[0:num_of_element]
        if st.session_state.flag == 0:
            step_df = df.iloc[st.session_state.i : st.session_state.size1 - 1]
        lines = plot_animation(step_df)
        line_plot = st.altair_chart(lines)
        line_plot = line_plot.altair_chart(lines)  
        return line_plot


def plotRep(df, size, start, num_of_element, line_plot):
        for i in range(start, num_of_element - size):  
                st.session_state.start=i 
                st.session_state.startSize = i-1
                step_df = df.iloc[i:size + i]
                st.session_state.size1 = size + i
                lines = plot_animation(step_df)
                line_plot.altair_chart(lines)
                time.sleep(.1)   #
        if st.session_state.size1 == num_of_element - 1:
            st.session_state.flag =1
            step_df = df.iloc[0:num_of_element]
            lines = plot_animation(step_df)
            line_plot.altair_chart(lines)


def show_plot(samples, samples_after_moidifcation, sampling_rate):
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
    common_df = df_afterUpload.merge(df_afterInverse, left_on='time', right_on='time_after')
    common_df.pop("time_after")
    num_of_element = common_df.shape[0]  # number of elements in the dataframe
    burst = 10  # number of elements (months) to add to the plot
    size = burst
    line_plot = currentState(common_df, size, num_of_element)
    plotRep(common_df, size, st.session_state.start, num_of_element, line_plot)

   
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
    D = librosa.stft(y)  # STFT of y
    # apply logarithm to cast amplitude to Decibels
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=title_of_graph)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    return plt.gcf()


