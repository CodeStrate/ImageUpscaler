import cv2
import streamlit as st
import numpy as np
from PIL import Image

from io import BytesIO

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
    st.info("Load an image to upscale...⬆️ `(Currently only JPG/PNG)`")
    st.stop()

st.markdown("`Image Loaded : {}`".format(upload_util.name))

extension_used = upload_util.name.split(".")[-1]

# st.write(extension_used) working

# tabs
traditional, deep_scaling, sharpen, about_me = st.tabs(['Use Traditional Upscaling 🔢', 'Use Deep Scaler 📐', 'Sharpen Image (Gaussian) 🫨', 'About Me 😶‍🌫️'])

# def sharpen_image(image):
#     with st.spinner('Sharpening your Image ... PLEASE WAIT!'):
#         img = Image.open(image)
#         img = np.

def download_image(image):
    pillow_image = Image.fromarray(image)
    buffered = BytesIO()
    pillow_image.save(buffered, format="PNG")
    
    return buffered.getvalue()

def upscale_bicubic(method, ratio):
    with st.spinner("Upscaling...PLEASE WAIT!"):
        img = Image.open(upload_util)
        img = np.array(img)
        if method == 'Fast':
            new_width = int(img.shape[1] * ratio)
            new_height = int(img.shape[0] * ratio)

            bicubic = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            st.success("Operation Completed! 🎊")
        else:
            start_time = time.time()
            dst = bicubic_interpolation(img, ratio, -1/2)
            end_time = time.time()
            bicubic = np.clip(dst, 0, 255).astype(np.uint8)
            st.success("Operation Completed! Took {:.4f} seconds 🎊".format(end_time - start_time))

        return bicubic

def deep_upscaler(path, model, ratio):
    with st.spinner('Deep Upscaling...PLEASE WAIT!'):
        sr_object = cv2.dnn_superres.DnnSuperResImpl_create()
        sr_object.readModel(path)
        sr_object.setModel(model, ratio)

        img = Image.open(upload_util)
        img = np.array(img)

        start_time = time.time()
        upscaled_image = sr_object.upsample(img)
        end_time = time.time()

        st.success('The Model took {:.4f} seconds 🎊'.format(end_time - start_time))

        return upscaled_image


def load_models_and_get_scales():
    models_dir = 'models'
    models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith(".pb"):
            model_path = os.path.join(models_dir, filename)
            model_name = filename.split('.')[0]
            prefix = model_name.split('_')[0]
            scale_str = model_name.split('_')[-1]
            scale = int(scale_str.replace('x', ''))
            if prefix not in models:
                models[prefix] = {'paths': [model_path], 'scales': [scale]}
            else:
                models[prefix]['paths'].append(model_path)
                models[prefix]['scales'].append(scale)
    return models
# traditional upscaler

with traditional:
    select_type = st.selectbox('Select upscaling Method', options=('Fast', 'Custom'))
    upscale_ratio = st.slider('Select Upscaling Ratio (Eg. 2x, 3x, 0.5x, etc.)', min_value=0.5, max_value=4.0, step=0.1, format="%.1f")
    if select_type == 'Fast':
        st.info('Fast Bicubic is based on OpenCV2 resize function with interpolation')
    else:
        st.info('Custom Bicubic is based on my own implementation of Bicubic Interpolation')

    if st.button('Upscale Image'):

        bicubic = upscale_bicubic(select_type, upscale_ratio)
        st.image(bicubic, caption='Resized using Bicubic.')
        if bicubic is not None:
            downloadable_image = download_image(bicubic)
            filename = upload_util.name.split('.')[0]
            st.markdown('---')
            st.download_button('Download as PNG', downloadable_image, file_name=f'Deep_{filename}.png', mime='image/png')
        



with deep_scaling:
    models_info = {
        "EDSR" : "Enhanced Deep Residual Networks for Single Image",
        "ESPCN" : "Efficient Sub-Pixel Convolutional Neural Network for Real Time Single Image/Video",
        "FSRCNN" : "Accelerated Super-Resolution Convolutional Neural Network",
        "LapSRN" : "Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks",
    }

    models = load_models_and_get_scales()
    # st.write(models)
    selected_model = st.selectbox('Select your Deep Upscaling Model', options=list(models.keys()))
    selected_upscale_ratio = st.selectbox('Select Upscaling Ratio', options=list(models[selected_model]['scales']))

    selected_model_index = models[selected_model]['scales'].index(selected_upscale_ratio)
    selected_model_path = models[selected_model]['paths'][selected_model_index]

    if selected_model:
        st.info(f'{models_info[selected_model]}')
        
    # st.write(selected_model_index, selected_model_path)
    if st.button('Deep Upscale'):

        upscaled_image = deep_upscaler(selected_model_path, selected_model.lower(), selected_upscale_ratio)
        st.image(upscaled_image, caption=f'Upscaled using {selected_model}')

        if upscaled_image is not None:
            downloadable_image = download_image(upscaled_image)
            filename = upload_util.name.split('.')[0]
            st.markdown('---')
            st.download_button('Download as PNG', downloadable_image, file_name=f'Deep_{filename}.png', mime='image/png')

    

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