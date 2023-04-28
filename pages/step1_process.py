import streamlit as st
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter as sg
from BaselineRemoval import BaselineRemoval as br


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def baseline(x, lambda_, order_):
    obj = br(x)
    return obj.ZhangFit(lambda_=lambda_, porder=order_)

def smooth(x, window, order):
    x = sg(x, window, order)
    return x

def resize(spec_df, mode):
    from scipy.interpolate import interp1d
    ref_wave = np.loadtxt('demos/demo.txt')
    if mode == 1:
        wave = ref_wave[:189, 0]
    else:
        wave = ref_wave[189:, 0]
    f = interp1d(spec_df.wavenumber, spec_df.intensity, kind='cubic')
    intensity = f(wave)
    spec_df = pd.DataFrame({'wavenumber': wave, 'intensity': intensity})
    return spec_df


def load_data(upload_file):
    spec = pd.read_csv(upload_file, delimiter='\t', header=None)
    spec.columns = ['wavenumber', 'intensity']
    st.session_state['raw_spec'] = spec
    return spec 


def upload_module(upload_file):
    if upload_file.name[-3:] != 'txt':
        st.error('Please upload a txt file')
    spec = load_data(upload_file)
    return spec 


def smooth_module(spec_df):
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['intensity'].copy()
    col1, col2, _, _, _ = st.columns(5)
    with col1:
        st.markdown('*Smooth*')
    with col2:
        skip_smooth = st.checkbox('Skip', key='smooth')

    if not skip_smooth:
        window_size = st.slider('smooth window size', 3, 13, 7)
        order = st.slider('smooth order', 1, 5, 3)
        if order >= window_size:
            st.error('order must be less than window size')
            st.stop()
        spec_df['processed'] = smooth(spec_df['processed'], window_size, order)
    return spec_df 


def baseline_module(spec_df):
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['intensity'].copy()
    col1, col2, _, _, _ = st.columns(5)
    with col1:
        st.markdown('*Baseline removal*')
    with col2:
        skip_baseline = st.checkbox('Skip', key='skip_baseline')
    if not skip_baseline:
        lambda_ = st.slider('lambda_', 1, 500, 100)
        order_ = st.slider('order_', 1, 4, 2)
        if order_ >= lambda_:
            st.error('order must be less than lambda')
            st.stop()
        spec_df['processed'] = baseline(spec_df['processed'], lambda_, order_)
        
    return spec_df 


def normalize_module(spec_df):
    # cut out spectrum with 800-1800 cm-1
    tmp_spec1 = spec_df[(spec_df.wavenumber >= 798) & (spec_df.wavenumber <= 1803)]
    if len(tmp_spec1) != 189:
        tmp_spec1 = resize(tmp_spec1, mode=1)

    # cut out spectrum with 2700-3200 cm-1
    tmp_spec2 = spec_df[(spec_df.wavenumber >= 2698) & (spec_df.wavenumber <= 3203)]
    if len(tmp_spec2) != 95:
        tmp_spec2 = resize(tmp_spec2, mode=2)

    spec_df = pd.concat([tmp_spec1, tmp_spec2])
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['intensity'].copy()
    spec_df['processed'] = minmax(spec_df['processed'])
    return spec_df


def run():
    raw_spec = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None

    st.subheader('Upload spectrum')
    st.write('you can use the demo spectrum we prepared for you')
    demo = st.selectbox('choose one demo', ['None', 'E.coli', "P.aeruginosa", "C.joostei", "E.faecalis", "S.aureus", "S.enterica"])
    if demo != 'None':
        raw_spec = load_data(f'demos/{demo}.txt')

    st.write('or you can upload your own spectrum')
    upload_file = st.file_uploader("Choose a file")    
    if upload_file:
        raw_spec = upload_module(upload_file)
    
    if raw_spec is not None:
        st.subheader('Pre-process', anchor='preprocess')
        
        spec = smooth_module(raw_spec)
        spec = baseline_module(spec)
        st.line_chart(data=spec, x='wavenumber')

        if st.button('Submit'):
            st.write('Your result is recorded in page 2: prediction')
            spec = normalize_module(spec)
            st.session_state['input_spec'] = spec
            st.sidebar.success('Click step 2 to get the prediction')

if __name__ == '__main__':
    run()