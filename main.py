import my_fun
import streamlit as st
import pandas as pd
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
    st.write("## Choose the mode ")
    Mode = st.selectbox(label="", options=[
                        'Frequency', 'Vowels', 'Music Instrument', 'Medical', 'Pitch Shift'])

    # print (file_uploaded.type )


########################### Main #######################################

def main():
    st.session_state.size1=0
    st.session_state.flag=0
    st.session_state.i=0
    st.session_state.start=0
    if file_uploaded:
        if file_uploaded.type == "audio/wav":

            magnitude_at_time, sample_rate = my_fun.to_librosa(file_uploaded)
            Data_frame_of_medical = pd.DataFrame()
            pitch_step = 0
            spectogram = st.sidebar.checkbox(label="Spectogram")

            st.session_state.show_spec = 0
            st.sidebar.write("## Audio before")
            st.sidebar.audio(file_uploaded.name)
            if spectogram:
                st.session_state.show_spec = 1
            else:
                st.session_state.show_spec = 0
#_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Modes Function_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _#

            if Mode == 'Frequency':
                dictnoary_values = {"0:1000": [0, 1000],
                                    "1000:2000": [1000, 2000],
                                    "3000:4000": [3000, 4000],
                                    "4000:5000": [4000, 5000],
                                    "5000:6000": [5000, 6000],
                                    "6000:7000": [6000, 7000],
                                    "7000:8000": [7000, 8000],
                                    "8000:9000": [8000, 9000],
                                    "9000:10000": [9000, 10000]
                                    }
                values_slider = [[0, 10, 1]]*len(list(dictnoary_values.keys()))

            elif Mode == 'Vowels':
                dictnoary_values = {"h": [1900, 5000],
                                    "R": [1500, 3000],
                                    "O": [500, 2000],
                                    "Y": [490, 2800]
                                    }
                values_slider = [[0, 10, 1]]*len(list(dictnoary_values.keys()))

            elif Mode == 'Music Instrument':
                dictnoary_values = {"Drum ": [0, 500],
                                    "Flute": [500, 1000],
                                    "Key": [1000, 2000],
                                    "Piano": [2000, 5000]
                                    }
                values_slider = [[0, 10, 1]]*len(list(dictnoary_values.keys()))

            elif Mode == 'Pitch Shift':
                dictnoary_values = {"Pitch Step": [0, 0]}
                values_slider = [[-12, 12, 1]]
                pitch_step = st.slider(label="Pitch Shift",
                                       max_value=12, min_value=-12, step=1, value=5)

            my_fun.processing(Mode, list(dictnoary_values.keys()), values_slider, magnitude_at_time,
                              sample_rate, st.session_state.show_spec, dictnoary_values, pitch_step)

        elif file_uploaded.type == "text/csv":
            if Mode == 'Medical':
                dictnoary_values = {"arrythmia 1": [0, 0],
                                    "arrythmia 2": [0, 0],
                                    "arrythmia 3": [0, 0],
                                    }
                values_slider = [[0.0, 2.0, 1]] * \
                    len(list(dictnoary_values.keys()))
                Data_frame_of_medical = pd.read_csv(file_uploaded)
                slider = my_fun.generate_slider(
                    list(dictnoary_values.keys()), values_slider, 0.1)
                my_fun.modifiy_medical_signal(Data_frame_of_medical, slider)


if __name__ == "__main__":
    main()
