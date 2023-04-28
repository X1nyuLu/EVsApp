from PIL import Image as Image
import streamlit as st

st.set_page_config(
    page_title="Raman for EVs",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

def intro():
    st.write("# Deep Learning-Enabled Raman Spectroscopic Identification of Extracellular Vesicles 👋")
    st.sidebar.success("Click **step 1** to start")
    st.subheader("introduction")
    toc = Image.open('toc.webp')
    st.image(toc, use_column_width=True,caption='TOC')
    st.markdown(
        """
        Pathogenic bacterial infections, exacerbated by increasing antimicrobial resistance, pose a major threat to human 
        health worldwide. Extracellular vesicles (EVs), secreted by bacteria and acting as their “long-distance weapons”, 
        play an important role in the occurrence and development of infectious diseases. However, no efficient methods to 
        rapidly detect and identify EVs of different bacterial origins are available. Here, label-free Raman spectroscopy 
        in combination with a new deep learning model of the attentional neural network (aNN) was developed to identify 
        pathogen-derived EVs at Gram±, species, strain, and even down to physiological levels. 
        Moreover, spectral markers underpinning EV discrimination were uncovered from subtly different EV spectra via 
        an interpretation algorithm of the integrated gradient. This developed platform provides an accurate and versatile 
        means to identify pathogen-derived EVs, spectral markers, and the biogenesis process. It will promote rapid diagnosis 
        and allow the timely treatment of bacterial infections.  

    """
    )

    st.subheader("Tutorial")
    st.markdown(
        """
        1. Upload your Raman spectrum in the ***step 1 process*** in the sidebar.
        2. Preprocess and click the "Submit" button.
        3. The result will be displayed in the ***step 2 prediction*** page.
        """)
    st.subheader("Citation")
    st.markdown(
        """
        here is [paper](https://pubs.acs.org/doi/10.1021/acs.analchem.2c02226) and [codes](https://10.26.15.61/xinyulu/EVs-Classification)  
        _____________________________________________________________

        if you find this app useful, please cite our paper:
        """
    )
    st.text(
        """
        @article{qin2022deep,
        title={Deep Learning-Enabled Raman Spectroscopic Identification of Pathogen-Derived Extracellular Vesicles and the Biogenesis Process},
        author={Qin, Yi-Fei and Lu, Xin-Yu and Shi, Zheng and Huang, Qian-Sheng and Wang, Xiang and Ren, Bin and Cui, Li},
        journal={Analytical Chemistry},
        volume={94},
        number={36},
        pages={12416--12426},
        year={2022},
        publisher={ACS Publications}
        }  
        """)
    

if __name__ == "__main__":
    intro()