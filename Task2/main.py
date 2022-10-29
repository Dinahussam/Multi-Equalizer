import functions
import pandas as pd
import streamlit as st  # data web app development
import streamlit_vertical_slider as svs
import wave, struct


# general styling and tab name
st.set_page_config(
    page_title="Equalizer",
    page_icon="✅",
    layout="wide",
)

# For any warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# styles from css file
with open(r"C:\Users\Function\Desktop\Task2\style.css") as design:
    st.markdown(f"<style>{design.read()}</style>", unsafe_allow_html=True)


# title
st.title("Equalizer")

# Save and upload
col_upload, col_save = st.columns([2, 2])

uploaded_file = col_upload.file_uploader('upload the Signal file', ['wav'], help='upload your Signal file')
if uploaded_file:
    df = pd.wave_read(uploaded_file)

with col_save:
    file_name = col_save.text_input('Write file name to be saved')
    if st.button('Save the current resulted Signal'):
        functions.save_signal(file_name)
        st.success("File is saved successfully as " + file_name + ".wav", icon="✅")


composer_cont = st.container()
col_check, col_figure = st.columns([1, 4])

with composer_cont:
    with col_check:
        st.markdown("## Signal")
        signal = st.radio(
            "",
            ('Frequency', 'Alphabets', 'Music Instrumentation', 'Medical Instrumentation'))

    with col_figure:

        if signal == 'Frequency':
            fig = functions.Functions.sin()
            fig2 = functions.Functions.layout_fig(fig)
            st.plotly_chart(fig2)

        if signal == 'Alphabets':
            sfig = functions.Functions.sin()
            fig2 = functions.Functions.layout_fig(fig)
            st.plotly_chart(fig2)

        if signal == 'Music Instrumentation':
            fig = functions.Functions.sin()
            fig2 = functions.Functions.layout_fig(fig)
            st.plotly_chart(fig2)

        if signal == 'Medical Instrumentation':
            fig = functions.Functions.sin()
            fig2 = functions.Functions.layout_fig(fig)
            st.plotly_chart(fig2)


def verticalSlider(key, min_value, max_value):
    svs.vertical_slider(
                        key=key,
                        step=1,
                        min_value=min_value,
                        max_value=max_value,
                        slider_color='#06283D',
                        track_color='lightgray',
                        thumb_color='#256D85',
                        )


# 10 Sliders
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
with col1:
    verticalSlider(1, 0, 100)
with col2:
    verticalSlider(2, 0, 100)
with col3:
    verticalSlider(3, 0, 100)
with col4:
    verticalSlider(4, 0, 100)
with col5:
    verticalSlider(5, 0, 100)
with col6:
    verticalSlider(6, 0, 100)
with col7:
    verticalSlider(7, 0, 100)
with col8:
    verticalSlider(8, 0, 100)
with col9:
    verticalSlider(9, 0, 100)
with col10:
    verticalSlider(10, 0, 100)