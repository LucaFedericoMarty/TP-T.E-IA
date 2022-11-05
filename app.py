import streamlit as st


with st.container():
    st.title("Trabajo Practico IA Deployement")
    st.write("Esta es la pagina de pruebas para nuestro IA")

thr = st.sidebar.slider("Detection Threshold", min_value = 0.0, max_value = 1.0, value = 0.3, step = 0.01)

# model = st.sidebar.selectbox("Select Model",  ({"EfficientDet0" : 'logs/model.tflite'}, {"EfficientDet1" : ': logs/model1.tflite'}))

image_file = st.file_uploader("Upload images for object detection", type=['png','jpeg'])

model = st.file_uploader("Model", type=['tflite'])

if image_file is not None:
    input_image = Image.open(image_file)
    st.image(input_image)

detect = st.button("Detect objects")

if detect:
    cwd = os.getcwd()
    # Change the test file path to your test image
    im = Image.open(image_file)
    im.thumbnail((512, 512), Image.ANTIALIAS)

    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="logs/model1.tflite")
    interpreter.allocate_tensors()

    # Run inference and draw detection result on the local copy of the original file
    detection_result_image = run_odt_and_draw_results(
      input_image,
      interpreter,
      threshold=thr
)

    # Show the detection result
    img = Image.fromarray(detection_result_image)
    st.image(img)
