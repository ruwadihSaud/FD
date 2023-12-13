import streamlit as st
import pickle
from sklearn.compose import make_column_transformer
import pandas as pd
import datetime
from PIL import Image, ImageDraw
import time
import base64
import sklearn
from sklearn.ensemble import RandomForestClassifier
import  streamlit_vertical_slider  as svs
import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
from xgboost import XGBClassifier

st.set_page_config(page_icon ="img\logo.png",page_title="FD", layout="wide", initial_sidebar_state="auto", menu_items=None)

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_background = get_img_as_base64("Picture1.png")

# css section
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img_background}");
background-size: 100%;
background-repeat: repeat-y;
background-position: top left; /*center*/
background-attachment: local;  /*fixed*/
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

#open css style file
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


#the page

selected = option_menu(None, ["Home","Uploaded File", 'About' , 'Creators'], 
        icons=['house', "cloud-upload", 'info-square-fill','people-fill'], menu_icon="cast", default_index=0, orientation="horizontal")


#'''Home page''' 
if selected == "Home":
    
    st.markdown('#')
    
    #the title of the page
    st.markdown('<h1 class="title">Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

    st.markdown('#')
    st.markdown('#')

    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14 = st.columns(14)
    #'''first row '''
    with col3:
        V4 =svs.vertical_slider(key="slider1", default_value=0, step=.5, min_value=-5.5, max_value=16.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V4 : {}</div>'.format(V4), unsafe_allow_html=True)

    with col4:
        V8 =svs.vertical_slider(key="slider2", default_value=0, step=.5, min_value=-73.0, max_value=20.0,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V8 : {}</div>'.format(V8), unsafe_allow_html=True)

    with col5:
        V10 =svs.vertical_slider(key="slider3", default_value=0, step=0.5, min_value=-24.5, max_value=23.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V10 : {}</div>'.format(V10), unsafe_allow_html=True)

    with col6:
        V12 =svs.vertical_slider(key="slider4", default_value=0, step=.5, min_value=-18.5, max_value=7.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V12 : {}</div>'.format(V12), unsafe_allow_html=True)

    with col7:
        V13 =svs.vertical_slider(key="slider5", default_value=0, step=.5, min_value=-5.5, max_value=7.0,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V13 : {}</div>'.format(V13), unsafe_allow_html=True)

    with col8:
        V14 =svs.vertical_slider(key="slider6", default_value=0, step=.5, min_value=-19.0, max_value=10.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V14 : {}</div>'.format(V14), unsafe_allow_html=True)

    with col9:
        V15 =svs.vertical_slider(key="slider7", default_value=0, step=.5, min_value=-4.0, max_value=8.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V15 : {}</div>'.format(V15), unsafe_allow_html=True)

    with col10:
        V17 =svs.vertical_slider(key="slider8", default_value=0, step=.5, min_value=-25.0, max_value=9.0,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V17 : {}</div>'.format(V17), unsafe_allow_html=True)

    with col11:
        V19 =svs.vertical_slider(key="slider9", default_value=0, step=.5, min_value=-7.0, max_value=5.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V19 : {}</div>'.format(V19), unsafe_allow_html=True)
    
    with col12:
        V25 =svs.vertical_slider(key="slider10", default_value=0, step=.5, min_value=-10.0, max_value=7.5,slider_color= '#1b4332',track_color='lightgray',thumb_color = '#52b788')
        st.markdown('<div class="sliders">V25 : {}</div>'.format(V12), unsafe_allow_html=True)

    st.markdown('#')

    col3,col4,col5 = st.columns([2, 3, 2])
    
    #'''second row '''
    with col4:
    
        st.markdown('#')
        
        col6,col7,col8 = st.columns([2, 3, 2])
        
        with col7:
            Amount = st.number_input('**Transaction Amount**', min_value=0.0,max_value=25691.0, value=0.0, step=.5, format="%f")
    
        st.markdown('#')
        st.markdown('#')
    
        button = st.button("**prediction**")

        dicts = {"V14":V14,"V17":V17,"V8":V8,"V10":V10,"V12":V12,"V4":V4,"V15":V15,'Amount': Amount,"V19":V19,"V25":V25,"V13":V13}  
        if button:
            
            #load model
            some = pickle.load(open("some-model","rb"))
            df = pd.DataFrame.from_dict([dicts])

            prediction = some.predict(df)
        
            if prediction==0:
                prediction = "Not Fraud"

            else:
                prediction = "Fraud"
            
            st.markdown('#')    
            col8 ,col9 ,col10 = st.columns([2, 3, 2])
            
            with col9: 
                st.markdown('<div class="prediction">The case is<br /> {} </div>'.format(prediction), unsafe_allow_html=True)


#'''about page'''
if selected == "About":
    
    # gif from global file
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://lottie.host/03853737-c5ac-4da0-adac-821a631e8a7f/n2ZScgj7Cq.json" 
    lottie_json = load_lottieurl(lottie_url)
    
    col1,col2 = st.columns(2) 
    
    with col1:
        st.markdown('<div class="about">Credit card fraud detection:</div>',unsafe_allow_html=True)
        st.markdown('<div class="about-info">strategies can vary depending on the credit card issuer. According to Inscribe, some of the most common practices involve using AI, machine learning and data analysis to review spending patterns and account behavior</div>',unsafe_allow_html=True)
    with col2:
        st_lottie(lottie_json,reverse=True, height=400, width=400, speed=0.5,)


#'''Creators page'''        
if selected == "Creators":
    
    st.markdown('#')
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col2:
        pf1 = get_img_as_base64("img\RSN.png")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf1}' class='img'>
                    <br />
                    <br />
                    RUWAIDAH SAUD
                    </div>""",unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    MARIAM AHMED
                    </div>""",unsafe_allow_html=True)
    
    with col4:
        pf3 = get_img_as_base64("img\Renad.jpg")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf3}' class='img'>
                    <br />
                    <br />
                    RENAD SAEED
                    </div>""",unsafe_allow_html=True)
    
    st.markdown('#')
    
    col6,col7,col8,col9,col10 = st.columns(5)
    
    with col7:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    SARA ALI
                    </div>""",unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    AHMED RAJA
                    </div>""",unsafe_allow_html=True)
    
    with col9:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    LUJAIN MAMDOUH
                    </div>""",unsafe_allow_html=True)
    
    st.markdown('#')
    
    col6,col7,col8,col9,col10 = st.columns(5)
    
    with col7:
        pf7 = get_img_as_base64("img\sruriq.jpg")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf7}' class='img'>
                    <br />
                    <br />
                    SHURUQ HASSAN
                    </div>""",unsafe_allow_html=True)
    
    with col8:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    FATIMA HASSAN
                    </div>""",unsafe_allow_html=True)
    
    with col9:
        st.markdown(f"""<div class='profile'>
                    <img src='https://t4.ftcdn.net/jpg/03/42/99/71/360_F_342997143_wz7x1yR7KWhmhSKF9OHwuQ2W4W7IUDvH.jpg' class='img'>
                    <br />
                    <br />
                    RAWAN ABDULLAH
                    </div>""",unsafe_allow_html=True)


#'''Uploaded File page''' 
if selected == "Uploaded File":
    
    st.markdown('#')
    
    st.markdown('<h1 class="title">Upload Your File</h1>', unsafe_allow_html=True)
    
    col1,col2,col3 = st.columns([2, 4, 2]) 
    
    with col2:
        
        st.markdown('#')
        uploaded_file = st.file_uploader("", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except:
                df = pd.read_excel(uploaded_file)

            st.markdown('<div class="sliders">Original File</div>', unsafe_allow_html=True)
            st.write(df.head(3))
            
            all = pickle.load(open("all-model","rb"))
            result = all.predict(df)
            
            st.markdown('#')
            
            st.markdown('<div class="sliders">Predict File</div>', unsafe_allow_html=True)
            df3 = pd.DataFrame({'Class': result})
            df3 = pd.concat([df3, df], axis = 1)
            st.write(df3.head(3))
            
            st.markdown('#')
            
            st.download_button('Download file',data=pd.DataFrame.to_csv(df3,index=False), mime='text/csv')
            
