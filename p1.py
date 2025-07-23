import streamlit as st
import pandas as pd
import requests
import os
import numpy as np
import joblib
from streamlit_lottie import st_lottie
loaded_model = joblib.load('crop_rec.joblib')
print(loaded_model.classes_)