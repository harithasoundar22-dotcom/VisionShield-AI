
import io
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="VisionShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# CUSTOM CSS - ORIGINAL DESIGN PRESERVED
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #f1f5fb;
    color: #172b4d;
}
.block-container {
    max-width: 1450px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
[data-testid="stHeader"] {
    background: #f1f5fb;
}
[data-testid="stSidebar"] {
    background: #14243d !important;
    min-width: 260px;
}
[data-testid="stSidebar"] > div {
    background: #14243d !important;
}
[data-testid="stSidebar"] * {
    color: #ffffff;
}
[data-testid="stSidebar"] .stButton button {
    width: 100%;
    min-height: 45px;
    text-align: left;
    background: #203653;
    color: #ffffff;
    border: 1px solid #304968;
    border-radius: 9px;
    transition: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #203653 !important;
    color: #ffffff !important;
    border-color: #304968 !important;
}
.brand {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
}
.brand-sub {
    color: #b9d5ff;
    font-size: 11px;
    letter-spacing: 1.3px;
    margin-top: 7px;
}
.hero {
    background: linear-gradient(120deg, #17345c, #2864b7);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 22px;
}
.hero h1 {
    color: #ffffff !important;
    font-size: 34px;
    margin: 0 0 8px 0;
}
.hero p {
    color: #e0ebff !important;
    font-size: 15px;
    margin: 0;
}
.section-title {
    color: #17345c;
    font-size: 23px;
    font-weight: 800;
    margin: 20px 0 10px 0;
}
.muted {
    color: #455b78;
    font-size: 14px;
}
.info-card {
    background: #ffffff;
    border: 1px solid #d8e3f2;
    border-radius: 14px;
    padding: 20px;
    min-height: 135px;
    margin-bottom: 12px;
}
.info-card h3 {
    color: #214b83 !important;
    font-size: 18px;
}
.info-card p {
    color: #344b68 !important;
    font-size: 14px;
    line-height: 1.7;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #d8e3f2;
    border-radius: 12px;
    padding: 16px;
}
div[data-testid="stMetricLabel"] {
    color: #344b68 !important;
}
div[data-testid="stMetricValue"] {
    color: #214b83 !important;
}
[data-testid="stFileUploader"] {
    background: #dceaff !important;
    border: 2px solid #8eb5f0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stFileUploader"] section {
    background: #f8faff !important;
    border: 1px solid #c4d8f6 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] button {
    background: #2858a5 !important;
    color: #ffffff !important;
    border: 1px solid #2858a5 !important;
    transition: none !important;
}
.stButton button {
    background: #2858a5;
    color: #ffffff;
    border: 1px solid #2858a5;
    border-radius: 9px;
    transition: none !important;
    box-shadow: none !important;
}
.stButton button:hover {
    background: #2858a5 !important;
    color: #ffffff !important;
    border-color: #2858a5 !important;
}
.stDownloadButton button {
    background: #137b82;
    color: #ffffff;
    border: 1px solid #137b82;
    border-radius: 10px;
    padding: 12px;
    width: 100%;
    font-weight: 700;
    transition: none !important;
    box-shadow: none !important;
}
.stDownloadButton button:hover {
    background: #137b82 !important;
    color: #ffffff !important;
    border-color: #137b82 !important;
}
.status-card {
    background-color: #e8f2ff;
    border: 2px solid #3678c8;
    border-radius: 12px;
    padding: 22px;
    margin: 15px 0;
}
.status-card h3 {
    color: #17457a !important;
    margin-top: 0;
    font-size: 20px;
}
.status-card p {
    color: #20344f !important;
    font-size: 16px;
    line-height: 1.8;
    margin-bottom: 0;
}
hr {
    border-color: #d8e3f2;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# IMAGE PROCESSING FUNCTIONS
# ==================================================
def analyse_image(
    rgb,
    filter_type="Gaussian",
    enhance=True,
    threshold_method="Otsu",
    canny_low=45,
    canny_high=120,
    min_area=150
):
    # Convert RGB image to grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Improve contrast using CLAHE
    if enhance:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )
        enhanced = clahe.apply(gray)
    else:
        enhanced = gray.copy()

    # Reduce noise
    if filter_type == "Gaussian":
        filtered = cv2.GaussianBlur(
            enhanced, (5, 5), 0
        )
    elif filter_type == "Median":
        filtered = cv2.medianBlur(enhanced, 5)
    else:
        filtered = enhanced.copy()

    # Image sharpness and brightness
    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    brightness = float(np.mean(gray))

    # Thresholding
    if threshold_method == "Otsu (automatic)":
        _, binary = cv2.threshold(
            filtered, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            filtered, 127, 255,
            cv2.THRESH_BINARY
        )

    # Morphological opening to remove small foreground noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel
    )

    # Canny edge detection
    edges = cv2.Canny(
        filtered, canny_low, canny_high
    )

    # Draw candidate regions from threshold image
    contours, _ = cv2.findContours(
        cleaned.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    outlined = rgb.copy()
    candidate_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            candidate_count += 1
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                outlined,
                (x, y),
                (x + w, y + h),
                (255, 65, 65),
                2
            )

    # Overlay edges on the original image
    edge_overlay = rgb.copy()
    edge_overlay[edges > 0] = [255, 65, 65]

    return {
        "gray": gray,
        "enhanced": enhanced,
        "filtered": filtered,
        "binary": binary,
        "cleaned": cleaned,
        "edges": edges,
        "overlay": edge_overlay,
        "outlined": outlined,
        "blur_score": blur_score,
        "brightness": brightness,
        "candidate_count": candidate_count
    }


def to_png_bytes(image_array):
    """Convert grayscale or RGB image arrays to PNG bytes."""
    output = io.BytesIO()

    if len(image_array.shape) == 2:
        image = Image.fromarray(image_array)
    else:
        image = Image.fromarray(image_array)

    image.save(output, format="PNG")
    return output.getvalue()


# ==================================================
# NAVIGATION STATE - ORIGINAL
# ==================================================
if "page" not in st.session_state:
    st.session_state["page"] = "Home"


def navigate(destination):
    st.session_state["page"] = destination


# ==================================================
# SIDEBAR - ORIGINAL
# ==================================================
with st.sidebar:
    st.markdown(
        '<div class="brand">🛡️ VisionShield AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">SMARTER VISION · SAFER METAL</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### MAIN MENU")

    pages = [
        ("🏠  Home", "Home"),
        ("🖼️  Image Analysis", "Image Analysis"),
        ("⚙️  Defect Classification", "Defect Classification"),
        ("ℹ️  About", "About")
    ]

    for label, destination in pages:
        selected = st.session_state["page"] == destination
        button_label = ("●  " if selected else "") + label

        st.button(
            button_label,
            key=f"nav_{destination}",
            on_click=navigate,
            args=(destination,),
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 🔍 Inspection System")
    st.caption("Computer vision for metal-surface image analysis.")
    st.caption("HackZen 2.0 · Project Prototype")


# ==================================================
# COMMON HEADER - ORIGINAL
# ==================================================
st.markdown("""
<div class="hero">
    <h1>VisionShield AI</h1>
    <p>Intelligent metal-surface image analysis using computer vision.</p>
</div>
""", unsafe_allow_html=True)

page = st.session_state["page"]


# ==================================================
# HOME - ORIGINAL
# ==================================================
if page == "Home":

    st.markdown(
        '<div class="section-title">Welcome to Your Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Explore the tools available in VisionShield AI.</p>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🖼️ Image Analysis</h3>
            <p>Upload an image to check blur, brightness, contrast and surface edges.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>⚙️ Defect Classification</h3>
            <p>Explore the planned machine-learning classification module.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
            <h3>📊 Quality Metrics</h3>
            <p>Review image dimensions and basic image-quality indicators.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🚀 Start Analysing")
    st.write(
        "Select Image Analysis from the left sidebar to upload "
        "and inspect a metal-surface photograph."
    )

    st.button(
        "🖼️ Open Image Analysis",
        key="home_open_analysis",
        on_click=navigate,
        args=("Image Analysis",),
        use_container_width=True
    )


# ==================================================
# IMAGE ANALYSIS - UPGRADED, ORIGINAL STYLE
# ==================================================
elif page == "Image Analysis":

    st.markdown(
        '<div class="section-title">📤 Upload Your Metal Surface Image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="muted">Select Browse files below. '
        'Supported formats: JPG, JPEG, PNG and BMP.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Browse files",
        type=["jpg", "jpeg", "png", "bmp"],
        key="metal_image_uploader",
        help="Choose a clear photograph of a metal sheet or component."
    )

    if uploaded_file is None:

        st.markdown("""
        <div class="info-card">
            <h3>📁 Ready for Image Upload</h3>
            <p>
            Choose an image using the Browse files button above.
            Your image and analysis results will appear here.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Available Analysis Tools")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🔎 Blur Detection</h3>
                <p>Calculates a basic image sharpness score.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>☀️ Brightness & Contrast</h3>
                <p>Measures brightness and offers contrast enhancement.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="info-card">
                <h3>〰️ Edge & Contour Detection</h3>
                <p>Highlights image edges and candidate regions.</p>
            </div>
            """, unsafe_allow_html=True)

    else:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            rgb = np.array(pil_image)

            st.markdown("### ⚙️ Processing Settings")

            setting1, setting2 = st.columns(2)

            with setting1:
                filter_type = st.selectbox(
                    "Noise filtering",
                    ["Gaussian", "Median", "None"]
                )

                enhance = st.checkbox(
                    "Enable contrast enhancement (CLAHE)",
                    value=True
                )

            with setting2:
                threshold_method = st.selectbox(
                    "Thresholding method",
                    ["Otsu (automatic)", "Fixed threshold (127)"]
                )

                min_area = st.slider(
                    "Minimum candidate region area",
                    min_value=10,
                    max_value=5000,
                    value=150,
                    step=10
                )

            st.markdown("### Edge Detection Settings")

            edge1, edge2 = st.columns(2)

            with edge1:
                canny_low = st.slider(
                    "Canny lower threshold",
                    0, 254, 45
                )

            with edge2:
                canny_high = st.slider(
                    "Canny upper threshold",
                    1, 255, 120
                )

            if canny_low >= canny_high:
                st.warning(
                    "The upper threshold must be greater than "
                    "the lower threshold."
                )
                st.stop()

            results = analyse_image(
                rgb,
                filter_type=filter_type,
                enhance=enhance,
                threshold_method=threshold_method,
                canny_low=canny_low,
                canny_high=canny_high,
                min_area=min_area
            )

            st.success("Image processed successfully!")

            st.markdown(
                '<div class="section-title">🖼️ Image Review</div>',
                unsafe_allow_html=True
            )

            original_col, processed_col = st.columns(2, gap="large")

            with original_col:
                st.markdown("#### Original Image")
                st.image(rgb, width="stretch")

            with processed_col:
                st.markdown("#### Edge Visualization")
                st.image(results["overlay"], width="stretch")
                st.caption(
                    "Red lines indicate detected edges, not confirmed defects."
                )

            st.markdown(
                '<div class="section-title">📊 Analysis Results</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric("Blur Score", f"{results['blur_score']:.1f}")
                st.caption("Lower scores can indicate blur.")

            with c2:
                brightness = results["brightness"]
                st.metric("Brightness", f"{brightness:.1f} / 255")

                if brightness < 45:
                    st.warning("Image may be too dark.")
                elif brightness > 220:
                    st.warning("Image may be too bright.")
                else:
                    st.success("Basic brightness check passed.")

            with c3:
                st.metric(
                    "Image Dimensions",
                    f"{rgb.shape[1]} × {rgb.shape[0]} px"
                )
                st.caption("Width × height")

            with c4:
                st.metric(
                    "Candidate Regions",
                    results["candidate_count"]
                )
                st.caption("Based on thresholding and contour area.")

            st.markdown(
                '<div class="section-title">🔬 Processing Results</div>',
                unsafe_allow_html=True
            )

            # Grayscale and enhanced image
            r1, r2 = st.columns(2, gap="large")

            with r1:
                st.markdown("#### 1. Grayscale Image")
                st.image(results["gray"], width="stretch")

            with r2:
                st.markdown("#### 2. Contrast-Enhanced Image")
                st.image(results["enhanced"], width="stretch")

            # Noise filtering and thresholding
            r3, r4 = st.columns(2, gap="large")

            with r3:
                st.markdown("#### 3. Noise-Filtered Image")
                st.image(results["filtered"], width="stretch")

            with r4:
                st.markdown("#### 4. Threshold Image")
                st.image(results["binary"], width="stretch")

            # Morphology and edges
            r5, r6 = st.columns(2, gap="large")

            with r5:
                st.markdown("#### 5. Noise-Cleaned Threshold")
                st.image(results["cleaned"], width="stretch")

            with r6:
                st.markdown("#### 6. Canny Edge Detection")
                st.image(results["edges"], width="stretch")

            st.markdown("#### 7. Candidate Region Highlighting")
            st.image(results["outlined"], width="stretch")

            st.warning(
                "Candidate regions are identified using thresholding "
                "and contour area. They are not confirmed metal defects. "
                "Lighting and surface texture can affect these results."
            )

            st.markdown("### 📝 Image Quality Summary")

            if results["blur_score"] < 70:
                st.warning(
                    "The image may be blurry. Try a sharper photograph."
                )
            else:
                st.success("No basic blur warning detected.")

            st.info(
                "This prototype performs image processing and "
                "candidate-region highlighting. It does not yet "
                "contain a trained defect-classification model."
            )

            st.markdown("### ⬇️ Download Processed Images")

            d1, d2 = st.columns(2)

            with d1:
                st.download_button(
                    "Download Edge Visualization",
                    data=to_png_bytes(results["overlay"]),
                    file_name="visionshield_edge_visualization.png",
                    mime="image/png",
                    key="download_edges",
                    use_container_width=True
                )

            with d2:
                st.download_button(
                    "Download Candidate Regions",
                    data=to_png_bytes(results["outlined"]),
                    file_name="visionshield_candidate_regions.png",
                    mime="image/png",
                    key="download_candidates",
                    use_container_width=True
                )

            st.download_button(
                "Download Threshold Image",
                data=to_png_bytes(results["binary"]),
                file_name="visionshield_threshold.png",
                mime="image/png",
                key="download_threshold",
                use_container_width=True
            )

        except Exception as error:
            st.error(f"Unable to process this image: {error}")


# ==================================================
# DEFECT CLASSIFICATION - ORIGINAL
# ==================================================
elif page == "Defect Classification":

    st.markdown(
        '<div class="section-title">⚙️ Defect Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>AI-Based Metal Surface Inspection</h3>
        <p>
        VisionShield AI supports metal-surface image inspection
        through computer vision and can be extended with machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-card">
        <h3>⚙️ Classification Module Status</h3>
        <p>
        Automatic defect classification is not connected yet.
        A trained model and labelled training images are required
        to predict defect categories reliably.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Planned Workflow")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="info-card">
            <h3>01 · Dataset Preparation</h3>
            <p>Collect and label images of normal and defective metal surfaces.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
            <h3>02 · Model Training</h3>
            <p>Train a machine-learning model using the labelled images.</p>
        </div>
        """, unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""
        <div class="info-card">
            <h3>03 · Model Evaluation</h3>
            <p>Evaluate using unseen images and suitable classification metrics.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="info-card">
            <h3>04 · Dashboard Integration</h3>
            <p>Connect an evaluated model to make predictions on uploaded images.</p>
        </div>
        """, unsafe_allow_html=True)


# ==================================================
# ABOUT - UPDATED FEATURES
# ==================================================
elif page == "About":

    st.markdown(
        '<div class="section-title">ℹ️ About VisionShield AI</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3>Project Overview</h3>
        <p>
        VisionShield AI is a computer-vision prototype for reviewing
        metal-surface photographs and visualizing image features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Current Features")

    st.write("- Metal-surface image upload")
    st.write("- Grayscale conversion")
    st.write("- CLAHE contrast enhancement")
    st.write("- Gaussian and median noise filtering")
    st.write("- Otsu and fixed thresholding")
    st.write("- Morphological noise cleaning")
    st.write("- Canny edge detection")
    st.write("- Contour-based candidate region highlighting")
    st.write("- Blur and brightness measurements")
    st.write("- Downloadable image results")

    st.markdown("### Technologies Used")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Python**")
        st.caption("Application logic")

    with c2:
        st.markdown("**OpenCV + NumPy**")
        st.caption("Image processing")

    with c3:
        st.markdown("**Streamlit + Pillow**")
        st.caption("Dashboard and image handling")


# ==================================================
# FOOTER - ORIGINAL
# ==================================================
st.markdown("---")

st.caption(
    "🛡️ VisionShield AI | HackZen 2.0 | Computer Vision Prototype"
)
