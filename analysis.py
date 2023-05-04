import streamlit as st

import json 
import numpy as np
import pandas as pd
import altair as alt

import torch 
from ANN import ANN

json_root = "label.json"
save_path = "EV_ann.pth"

net = ANN(n_classes=6)
net.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
st.session_state['net'] = net


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def scale(x):
    x = np.abs(x)
    x = x.mean(0).flatten()
    x = (x - x.min()) / (x.max() - x.min())
    return x


def prediction():
    st.subheader('Prediction')
    inputs = torch.tensor(st.session_state['input_spec']['processed'].to_numpy(), dtype=torch.float32).reshape(1, 1, -1)

    with open(json_root) as f:
        st.session_state['class_indict'] = json.load(f)

    print("Start to test")
    net.eval()
    with torch.no_grad():
        with st.spinner('Wait for it...'):
            outputs = net(inputs)
            outputs = torch.softmax(outputs, dim=-1)
            prob, pred = torch.max(outputs, dim=1)

            outputs = outputs.numpy()
            prob = prob.numpy()
            pred = pred.numpy()

    res = pd.DataFrame({'label':st.session_state['class_indict'].values(), 'confidence':outputs.flatten()*100})
    st.bar_chart(data=res, x='label')
    pred_label = st.session_state['class_indict'][f'{int(pred)}']
    st.markdown(f'the spectrum you input is ***{pred_label}*** with confidence ***{prob[0]*100:.2f}%***')
    return pred_label


def analysis(pred_label):

    st.subheader('Attribution Analysis')

    from captum.attr import IntegratedGradients

    test_data = torch.tensor(st.session_state['input_spec']['processed'].to_numpy(), dtype=torch.float32).reshape(1, 1, -1)
    wave = st.session_state['input_spec']['wavenumber'].to_numpy()

    ig = IntegratedGradients(net)
    net.eval()
    net.zero_grad()

    attr = ig.attribute(test_data, target=0, baselines=0)
    attr = attr.detach().numpy()

    intensity = minmax(test_data.flatten()[:189])
    attr = scale(attr[:, :, :189])

    df = pd.DataFrame({'wavenumber': wave[:189], 'intensity': intensity, 'attribution': -attr})

    # Create the line chart
    line_chart = alt.Chart(df).mark_line().encode(
        x=alt.X('wavenumber:Q'),
        y=alt.Y('intensity:Q')
    )

    # Create the bar chart
    bar_chart = alt.Chart(df).mark_bar(color='orange').encode(
        x=alt.X('wavenumber:Q'),
        y=alt.Y('attribution:Q'),
        tooltip=[alt.Tooltip('wavenumber:Q'), alt.Tooltip('attribution:Q')]
    )

    # Combine the two charts into one figure
    chart = line_chart + bar_chart

    # Add the zoom selection tool
    chart = chart.add_selection(
        alt.selection_interval(bind='scales', encodings=['x'])
    )
    st.altair_chart(chart, use_container_width=True)

    from config import contribution

    target_band = wave[np.argmax(attr)]
    st.markdown('The most important Raman band is around %.2f $cm^{-1}$' %(target_band))
    
    bands = contribution[f'{pred_label}']
    target_contribution = sorted(bands.items(), key=lambda x: abs(x[0]-target_band))[0]
    st.markdown('This band is mainly contributed by ***%s***' %(target_contribution[1]))
    
def test():
    pred_label = prediction()
    analysis(pred_label)
    st.markdown(
        """
        ---  
        This APP is still inder development.
        If you have any questions, please click [here](mailto:luxinyu@stu.xmu.edu.cn) to contact me :smile:
        """
    )
if __name__ == '__main__':
        
    if 'input_spec' in st.session_state:
        test()
    else:
        st.warning('Please upload and submit a spectrum first')