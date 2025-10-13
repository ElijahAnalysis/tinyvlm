import streamlit as st
import requests
import io
import base64
from typing import List, Tuple, Optional
from PIL import Image

# ===== CONFIGURATION =====
API_URL = "http://159.89.15.19:8000"
CATALOG_SIZE = 50
GRID_COLUMNS = 4
RESULTS_PER_PAGE = 20

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 45px;
        font-weight: 500;
    }
    
    img {
        border-radius: 8px;
        transition: transform 0.2s, box-shadow 0.2s;
        cursor: pointer;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .similarity-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ===== SESSION STATE INITIALIZATION =====
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'page': 'home',
        'search_results': [],
        'catalog_images': [],
        'selected_image': None,
        'search_query': '',
        'last_search_type': None,
        'catalog_loaded': False,
        'image_cache': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ===== IMAGE LOADING FUNCTIONS =====
@st.cache_data(ttl=3600, show_spinner=False)
def load_image_as_base64(url: str) -> Optional[str]:
    """Load image from URL and convert to base64 for display"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            # Resize if too large to improve loading
            if img.width > 800 or img.height > 800:
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        st.error(f"Failed to load image: {str(e)}")
    return None


def load_image_from_url(url: str) -> Optional[Image.Image]:
    """Load PIL Image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
    except:
        pass
    return None


# ===== API FUNCTIONS =====
def check_api_status() -> bool:
    """Check if API is responsive"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        return response.status_code == 200
    except:
        return False


def fetch_all_images() -> List[str]:
    """Fetch complete image list from API"""
    try:
        response = requests.get(f"{API_URL}/list_images", timeout=15)
        if response.status_code == 200:
            images = response.json().get('images', [])
            st.success(f"✅ Loaded {len(images)} images from API")
            return images
        else:
            st.error(f"API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to fetch images: {str(e)}")
    return []


def search_by_text(query: str, top_k: int = 20) -> List[Tuple[str, float]]:
    """Search images by text description"""
    try:
        response = requests.post(
            f"{API_URL}/encode_text_query",
            params={"text_query": query, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            results = response.json().get('top_k_similar_items', [])
            return [(item['url'], item['similarity']) for item in results]
        else:
            st.error(f"Search failed with status: {response.status_code}")
    except Exception as e:
        st.error(f"Search error: {str(e)}")
    return []


def search_by_image(image_bytes: io.BytesIO, top_k: int = 20) -> List[Tuple[str, float]]:
    """Search images by uploaded image"""
    try:
        image_bytes.seek(0)
        files = {"image_query": ("query.jpg", image_bytes, "image/jpeg")}
        response = requests.post(
            f"{API_URL}/encode_image_query",
            files=files,
            params={"top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            results = response.json().get('top_k_similar_items', [])
            return [(item['url'], item['similarity']) for item in results]
        else:
            st.error(f"Image search failed with status: {response.status_code}")
    except Exception as e:
        st.error(f"Image search error: {str(e)}")
    return []


def find_similar_images(image_url: str, top_k: int = 21) -> List[Tuple[str, float]]:
    """Find similar images to a given URL"""
    try:
        img_response = requests.get(image_url, timeout=10)
        if img_response.status_code == 200:
            image_bytes = io.BytesIO(img_response.content)
            results = search_by_image(image_bytes, top_k)
            return [(url, score) for url, score in results if url != image_url][:20]
        else:
            st.error(f"Failed to fetch image from URL: {img_response.status_code}")
    except Exception as e:
        st.error(f"Error finding similar images: {str(e)}")
    return []


# ===== UI COMPONENTS =====
def render_header():
    """Render page header with navigation"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# 🔍 Image Search Engine")
        st.markdown("##### Powered by Visual AI")
    
    with col3:
        if check_api_status():
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")


def render_search_box():
    """Render search interface"""
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔤 Text Search", "📸 Image Search"])
    
    with tab1:
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Search",
                placeholder="Describe what you're looking for...",
                label_visibility="collapsed",
                key="text_search_input"
            )
        with col2:
            search_btn = st.button("Search", key="text_search_btn", type="primary")
        
        if search_btn and query:
            with st.spinner("Searching..."):
                results = search_by_text(query, RESULTS_PER_PAGE)
                if results:
                    st.session_state.search_results = results
                    st.session_state.search_query = query
                    st.session_state.last_search_type = 'text'
                    st.session_state.page = 'results'
                    st.rerun()
                else:
                    st.warning("No results found. Try different keywords.")
    
    with tab2:
        uploaded_file = st.file_uploader(
            "Upload an image to find similar ones",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="visible"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_file, caption="Your image", use_container_width=True)
            with col2:
                if st.button("Find Similar Images", type="primary", key="image_search_btn"):
                    with st.spinner("Analyzing image..."):
                        results = search_by_image(uploaded_file, RESULTS_PER_PAGE)
                        if results:
                            st.session_state.search_results = results
                            st.session_state.search_query = "Image-based search"
                            st.session_state.last_search_type = 'image'
                            st.session_state.page = 'results'
                            st.rerun()
                        else:
                            st.warning("No similar images found.")


def render_image_grid(images: List, show_scores: bool = False):
    """Render images in responsive grid using base64 encoding"""
    if not images:
        st.info("No images to display")
        return
    
    cols_per_row = GRID_COLUMNS
    
    # Show loading progress
    if len(images) > 8:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, col in enumerate(cols):
            if i + idx < len(images):
                item = images[i + idx]
                url, score = (item[0], item[1]) if isinstance(item, tuple) else (item, None)
                
                with col:
                    # Load image as base64
                    img_data = load_image_as_base64(url)
                    
                    if img_data:
                        st.markdown(
                            f'<img src="{img_data}" style="width:100%; border-radius:8px;">',
                            unsafe_allow_html=True
                        )
                        
                        if show_scores and score is not None:
                            st.markdown(
                                f'<div class="similarity-badge">Match: {score:.1%}</div>',
                                unsafe_allow_html=True
                            )
                        
                        if st.button("View Details", key=f"view_{i}_{idx}_{hash(url)}", use_container_width=True):
                            st.session_state.selected_image = url
                            st.session_state.page = 'detail'
                            st.rerun()
                    else:
                        st.error("❌ Failed to load")
                        with st.expander("Details"):
                            st.code(url[:80] + "...")
        
        # Update progress
        if len(images) > 8:
            progress = min(1.0, (i + cols_per_row) / len(images))
            progress_bar.progress(progress)
            status_text.text(f"Loading images... {int(progress * 100)}%")
    
    # Clear progress indicators
    if len(images) > 8:
        progress_bar.empty()
        status_text.empty()


# ===== PAGE VIEWS =====
def show_home_page():
    """Display home page with search and catalog"""
    render_header()
    render_search_box()
    
    st.markdown("---")
    st.markdown("### 📚 Browse Catalog")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.catalog_images = []
            st.session_state.catalog_loaded = False
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("🏠 Load Catalog", use_container_width=True):
            st.session_state.catalog_images = []
            st.session_state.catalog_loaded = False
            st.rerun()
    
    # Load catalog
    if not st.session_state.catalog_loaded:
        with st.spinner("Loading catalog from API..."):
            all_images = fetch_all_images()
            if all_images:
                import random
                st.session_state.catalog_images = random.sample(
                    all_images, 
                    min(CATALOG_SIZE, len(all_images))
                )
                st.session_state.catalog_loaded = True
                st.info(f"📸 Displaying {len(st.session_state.catalog_images)} random images")
            else:
                st.error("❌ Failed to load catalog. Please check API connection.")
                return
    
    if st.session_state.catalog_images:
        render_image_grid(st.session_state.catalog_images, show_scores=False)
    else:
        st.warning("No images loaded. Click 'Load Catalog' to fetch images.")


def show_results_page():
    """Display search results"""
    st.markdown("# 🔍 Search Results")
    
    if st.button("← Back to Home", key="back_home"):
        st.session_state.page = 'home'
        st.rerun()
    
    st.markdown(f"**Query:** {st.session_state.search_query}")
    st.markdown(f"**Results:** {len(st.session_state.search_results)} images found")
    st.markdown("---")
    
    if st.session_state.search_results:
        render_image_grid(st.session_state.search_results, show_scores=True)
    else:
        st.info("No results to display")


def show_detail_page():
    """Display detailed view with similar images"""
    st.markdown("# 📸 Image Details")
    
    if st.button("← Back", key="back_detail"):
        st.session_state.page = 'results' if st.session_state.search_results else 'home'
        st.rerun()
    
    image_url = st.session_state.selected_image
    
    if image_url:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            img_data = load_image_as_base64(image_url)
            if img_data:
                st.markdown(
                    f'<img src="{img_data}" style="width:100%; border-radius:8px;">',
                    unsafe_allow_html=True
                )
            else:
                st.error("Failed to load image")
            
            with st.expander("🔗 Image URL"):
                st.code(image_url)
        
        with col2:
            st.markdown("### 🔗 Find Similar Images")
            if st.button("Search for Similar", type="primary", use_container_width=True):
                with st.spinner("Finding similar images..."):
                    similar = find_similar_images(image_url, 21)
                    if similar:
                        st.session_state.search_results = similar
                        st.session_state.search_query = "Similar images"
                        st.session_state.page = 'results'
                        st.rerun()
                    else:
                        st.warning("No similar images found")
        
        st.markdown("---")
        st.markdown("### 🎯 Similar Items")
        
        with st.spinner("Loading similar images..."):
            similar_images = find_similar_images(image_url, 13)
            if similar_images:
                render_image_grid(similar_images[:12], show_scores=True)
            else:
                st.info("No similar images found")


# ===== MAIN APPLICATION =====
def main():
    init_session_state()
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        show_home_page()
    elif st.session_state.page == 'results':
        show_results_page()
    elif st.session_state.page == 'detail':
        show_detail_page()


if __name__ == "__main__":
    main()