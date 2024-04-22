import cv2
import streamlit as st

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

# traditional upscaler

with traditional:
    select_type = st.selectbox('Select upscaling Method', options=('Fast', 'Custom'))
    upscale_ratio = st.slider('Select Upscaling Ratio (Eg. 2x, 3x, 0.5x, etc.)', min_value=0.5, max_value=8.0, step=0.1, format="%.1f")


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