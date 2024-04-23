import cv2
import streamlit as st
import numpy as np
from PIL import Image

import time, os

from st_bicubic import bicubic_interpolation

st.set_page_config(
    page_title="Deep.Imager",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/hardik-sharma-0256cs',
        'Report a bug': "https://www.github.com/CodeStrate",
        'About': "## An Image Upscaler using Interpolation and Deep Learning..."
    }
)

global upload_util

st.title("Deep.Imager - Easy to use Image upscaler")
st.markdown("### Dashboard 🔲")

with st.sidebar:
    st.header("Load your Image here! 🖼️")

@st.cache_data
def load_image(input):
    image = cv2.imread(input)
    return image

upload_util = st.sidebar.file_uploader("Choose your Image!")

# sidebar ui elements
with st.sidebar:
    st.markdown("---")

with st.sidebar.popover("Traditional Upscaling"):
    st.write("1. Fast Bicubic\n 2. Custom Bicubic (My implementation)\n")

with st.sidebar.popover("Deep Upscaling"):
    st.write("1. EDSR\n 2. ESPCN\n 3. FSRCNN\n 4. LapSRN\n")

if upload_util is None:
    st.info("Load an image to upscale...⬆️")
    st.stop()

st.markdown("`Image Loaded : {}`".format(upload_util.name))

input_image = upload_util.name

extension_used = upload_util.name.split(".")[-1]

# st.write(extension_used) working

# tabs
traditional, deep_scaling, about_me = st.tabs(['Use Traditional Upscaling 🔢', 'Use Deep Scaler 📐', 'About Me 😶‍🌫️'])


def upscale_my_bicubic(image, ratio):
    with st.spinner("Upscaling...PLEASE WAIT!"):

        dst = bicubic_interpolation(image, ratio, -1/2)
        dst_normalized = np.clip(dst, 0, 255).astype(np.uint8)
        st.success("Operation Completed! 🎊")

        return dst_normalized



def load_models_and_get_scales():
    models_dir = 'models'
    models = {}
    scales = set()
    for filename in os.listdir(models_dir):
        if filename.endswith(".pb"):
            model_path = os.path.join(models_dir, filename)
            model_name = filename.split('.')[0]
            scale_str = model_name.split('_')[-1]
            scale = int(scale_str.replace('x', ''))
            models[model_name] = model_path
            scales.add(scale)
    return models, sorted(list(scales))
# traditional upscaler

with traditional:
    select_type = st.selectbox('Select upscaling Method', options=('Fast', 'Custom'))
    upscale_ratio = st.slider('Select Upscaling Ratio (Eg. 2x, 3x, 0.5x, etc.)', min_value=0.5, max_value=4.0, step=0.1, format="%.1f")
    if select_type == 'Fast':
        st.info('Fast Bicubic is based on OpenCV2 resize function with interpolation')
    else:
        st.info('Custom Bicubic is based on my own implementation of Bicubic Interpolation')

    if st.button('Upscale Image'):
        if select_type == 'Fast':

            img = Image.open(upload_util)
            img = np.array(img)
            new_width = int(img.shape[1] * upscale_ratio)
            new_height = int(img.shape[0] * upscale_ratio)

            bicubic = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            st.image(bicubic, caption='Resultant Image')
        else:
            img = Image.open(upload_util)
            img = np.array(img)

            bicubic = upscale_my_bicubic(img, upscale_ratio)

            st.image(bicubic, caption='Resized using My Bicubic.')



with deep_scaling:
    models, scales = load_models_and_get_scales()
    st.write(models)
    st.write(scales)

# about me

with about_me:
    st.title('About Me! ❤️')
    st.header('Hardik Sharma')
    st.markdown("1. 🎮 Gamer\n 2. 🎨 Artist Part-time\n 3. 🖥️ Developer Full-time.\n 4. Pursuing AI and Machine Learning academically.\n")
    st.markdown('''
            * **`GitHub`** 🀄
                https://github.com/CodeStrate
            * **`LinkedIn`**  🔗 
                https://linkedin.com/in/hardik-sharma-0256cs
            * **`Portfolio`** 🖌️
                https://tinyurl.com/Cspfdr025
        ''')