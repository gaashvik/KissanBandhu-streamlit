import requests
import os
import streamlit as st
from streamlit_lottie import st_lottie
import mysql.connector as sql
custom_style = """
<style>
    .outlined-input input {
        border: 2px solid #000000;
        border-radius: 5px;
        padding: 8px;
    }
</style>
"""
conn=sql.connect(host="localhost",user="root",passwd="root",database="plant")
c=conn.cursor()
st.markdown(custom_style, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    else:
        return r.json()

rty,rtt=st.columns([3,2])
with rty:
    original_title= '<p style="font-family:American Captain; color:#038E6A; -webkit-text-stroke: 2px black; font-size: 100px;">AGRIFARM</p>'
    st.markdown(original_title, unsafe_allow_html=True)
with rtt:
    lottie_url = "https://lottie.host/608f7c65-2ba1-49a5-a0a2-7bd818fef95d/rkHZujVW64.json"
    you= load_lottieurl(lottie_url)
    st_lottie(
        you,
        speed=1,
        loop=True,
        quality="medium",
        )


lottie_url = "https://assets4.lottiefiles.com/packages/lf20_5zYhWw.json"
lottie_json = load_lottieurl(lottie_url)
nav=st.sidebar.radio("Navigation",["Home"])
if nav=="Home":
    tab1,tab2=st.tabs(["Login","Register "])
    with tab1:
        user,pw=st.columns(2)
        b=user.text_input("USERNAME",key="outlined-input")
        a=pw.text_input("PASSWORD",type="password")
        st.button("Submit")
        if a=='root' and b=='admin':
            cons= '<p style="font-family:Century Gothic; color:green; font-size: 20px;">CONNECTED SUCCESSFULLY!</p>'
            st.markdown(cons, unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(' ')
            with col2:
                st_lottie(
                lottie_json,
                speed=1,
                loop=True,
                height=200,
                width=200,
                quality="medium",
                )
            with col3:
                st.write(' ')
            opt=st.sidebar.button("Predict Disease")
            zt=st.sidebar.button("Recommend Crops")
            qw=st.sidebar.button("Predict Fertilizer")
            if opt:
                os.system(('streamlit run disease_pred_model.py'))
            elif zt:
                os.system(('cmd/c "streamlit run crop_rec_model.py"'))
            elif qw:
                    os.system(('cmd/c "streamlit run fertilizer_pred_model.py"'))
        else:
            login_query = "SELECT * FROM user_info WHERE email_id = %s AND passwd = %s"
            c.execute(login_query, (b, a))
            result = c.fetchone()
            if result:
                st.markdown('<p style="font-family:Century Gothic; color:green; font-size: 20px;">CONNECTED SUCCESSFULLY!</p>', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(' ')
                with col2:
                    st_lottie(
                    lottie_json,
                    speed=1,
                    loop=True,
                    height=200,
                    width=200,
                    quality="medium",
                    )
                with col3:
                    st.write(' ')
                opt=st.sidebar.button("Predict Disease")
                zt=st.sidebar.button("Recommend Crops")
                qw=st.sidebar.button("Predict Fertilizer")
                if opt:
                    os.system(('streamlit run disease_pred_model.py'))
                elif zt:
                    os.system(('cmd/c "streamlit run crop_rec_model.py"'))
                elif qw:
                     os.system(('cmd/c "streamlit run fertilizer_pred_model.py"'))
            else:
                cons= '<p style="font-family:Century Gothic; color:red; font-size: 20px;">WRONG PASSWORD OR USERNAME.</p>'
                st.markdown(cons, unsafe_allow_html=True)
                ll=load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_nvdh1vu4.json")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(' ')
                with col2:
                    st_lottie(
                        ll,
                    speed=1,
                    loop=False,
                    height=200,
                    width=200,
                    quality="medium",
                    )
                with col3:
                    st.write(' ')
with tab2:
        with st.form("registration_form"):
        # Add registration form fields
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            phone_number = st.text_input("Phone Number")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            dob = st.date_input("Date of Birth")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Register"):
                
                insert_query = """
                INSERT INTO user_info (f_name, l_name, phone_no, email_id, passwd, dob, gender)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                user_data = (first_name, last_name, str(phone_number), email, password, dob, gender)
                c.execute(insert_query, user_data)
                conn.commit()
                st.success("Registration Successful!")


            # cons= '<p style="font-family:Century Gothic; color:red; font-size: 20px;">WRONG PASSWORD OR USERNAME.</p>'
            # st.markdown(cons, unsafe_allow_html=True)
            # ll=load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_nvdh1vu4.json")
            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.write(' ')
            # with col2:
            #     st_lottie(
            #         ll,
            #     speed=1,
            #     loop=False,
            #     height=200,
            #     width=200,
            #     quality="medium",
            #     )
            # with col3:
            #     st.write(' ')
        
    

