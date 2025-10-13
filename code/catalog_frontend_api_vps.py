import streamlit as st
import requests
from PIL import Image
import io
import random

# ===== CONFIG =====
API_URL = "http://159.89.15.19:8000"  # Your VPS API endpoint
RANDOM_CATALOG_SIZE = 50

# ===== SESSION STATE =====
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'catalog'  # 'catalog', 'search_results', 'detail'
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'random_images' not in st.session_state:
    st.session_state.random_images = []


# ===== HELPER FUNCTIONS =====
def fetch_image_list():
    """Fetch all image URLs from backend."""
    try:
        response = requests.get(f"{API_URL}/list_images", timeout=10)
        if response.status_code == 200:
            return response.json().get('images', [])
        else:
            st.error(f"API Error: {response.status_code}")
            return []
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API for image list.")
        return []


def load_random_images(count=RANDOM_CATALOG_SIZE):
    """Select random images from the backend list."""
    all_images = fetch_image_list()
    if len(all_images) > count:
        return random.sample(all_images, count)
    return all_images


def search_by_text(text_query):
    """Search images via text query."""
    try:
        response = requests.post(f"{API_URL}/encode_text_query", params={"text_query": text_query}, timeout=30)
        if response.status_code == 200:
            results = response.json().get('top_k_similar_items', [])
            # Convert dict format to tuple format (url, score)
            return [(item['url'], item['similarity']) for item in results]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException:
        st.error("❌ Cannot connect to API.")
        return []


def search_by_image(image_file):
    """Search images via uploaded image."""
    try:
        image_file.seek(0)
        files = {"image_query": ("image.jpg", image_file, "image/jpeg")}
        response = requests.post(f"{API_URL}/encode_image_query", files=files, timeout=30)
        if response.status_code == 200:
            results = response.json().get('top_k_similar_items', [])
            # Convert dict format to tuple format (url, score)
            return [(item['url'], item['similarity']) for item in results]
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Cannot connect to API: {e}")
        return []


def display_image_grid(image_list, cols=3):
    """Display images in a grid layout."""
    rows = [image_list[i:i+cols] for i in range(0, len(image_list), cols)]
    for row in rows:
        columns = st.columns(cols)
        for idx, col in enumerate(columns):
            if idx < len(row):
                item = row[idx]
                if isinstance(item, tuple):
                    img_url, score = item
                else:
                    img_url, score = item, None
                try:
                    with col:
                        st.image(img_url, use_column_width=True)
                        st.caption(img_url.split("/")[-1])
                        if score is not None:
                            st.caption(f"Score: {score:.3f}")
                        if st.button("View Details", key=f"btn_{img_url}"):
                            st.session_state.selected_image = img_url
                            st.session_state.view_mode = 'detail'
                            st.rerun()
                except Exception as e:
                    col.error(f"Error loading image: {e}")


# ===== PAGES =====
def show_catalog():
    """Main catalog page."""
    st.title("🖼️ Image Catalog & Search")
    
    # API status
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API responding but unclear")
    except:
        st.error("❌ API Not Connected")
    
    st.subheader("Search")
    col1, col2 = st.columns(2)
    
    # Text search
    with col1:
        text_query = st.text_input("🔍 Search by text", placeholder="Enter description...")
        if st.button("Search by Text", use_container_width=True):
            if text_query:
                results = search_by_text(text_query)
                if results:
                    st.session_state.search_results = results[:10]
                    st.session_state.view_mode = 'search_results'
                    st.rerun()
                else:
                    st.warning("No results found")
    
    # Image search
    with col2:
        uploaded_file = st.file_uploader("📸 Search by image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            results = search_by_image(uploaded_file)
            if results:
                st.session_state.search_results = results[:10]
                st.session_state.view_mode = 'search_results'
                st.rerun()
            else:
                st.warning("No results found")
    
    st.divider()
    
    # Random catalog
    st.subheader("Browse Catalog")
    if st.button("🔄 Refresh Catalog"):
        st.session_state.random_images = load_random_images(RANDOM_CATALOG_SIZE)
    if not st.session_state.random_images:
        st.session_state.random_images = load_random_images(RANDOM_CATALOG_SIZE)
    
    display_image_grid(st.session_state.random_images, cols=3)


def show_search_results():
    """Display search results."""
    st.title("🔍 Search Results")
    if st.button("← Back to Catalog"):
        st.session_state.view_mode = 'catalog'
        st.rerun()
    st.divider()
    if st.session_state.search_results:
        display_image_grid(st.session_state.search_results, cols=3)
    else:
        st.info("No results found.")


def show_detail():
    """Detail view of selected image with similar items."""
    st.title("📸 Image Details")
    if st.button("← Back"):
        st.session_state.view_mode = 'search_results' if st.session_state.search_results else 'catalog'
        st.rerun()
    
    selected_img = st.session_state.selected_image
    if selected_img:
        st.image(selected_img, use_column_width=True)
        st.subheader(selected_img.split("/")[-1])
        st.divider()
        st.subheader("🔗 Similar Items")
        
        try:
            # Fetch the image from URL
            img_response = requests.get(selected_img, timeout=10)
            if img_response.status_code != 200:
                st.error(f"Failed to fetch image: HTTP {img_response.status_code}")
                return
            
            # Create BytesIO object from image content
            image_bytes = io.BytesIO(img_response.content)
            
            # Search for similar images
            similar_items = search_by_image(image_bytes)
            
            if similar_items:
                # Filter out the selected image and take top 5
                similar_items = [
                    item for item in similar_items
                    if item[0] != selected_img
                ][:5]
                
                if similar_items:
                    display_image_grid(similar_items, cols=3)
                else:
                    st.info("No other similar images found.")
            else:
                st.info("No similar images found.")
        except Exception as e:
            st.error(f"Error fetching similar items: {str(e)}")


# ===== MAIN =====
def main():
    st.set_page_config(page_title="Image Search & Catalog", page_icon="🖼️", layout="wide")
    
    # CSS tweaks
    st.markdown("""
        <style>
        .stButton>button { width: 100%; }
        div[data-testid="stImage"] { transition: transform 0.2s; }
        div[data-testid="stImage"]:hover { transform: scale(1.02); }
        </style>
    """, unsafe_allow_html=True)
    
    # View routing
    if st.session_state.view_mode == 'catalog':
        show_catalog()
    elif st.session_state.view_mode == 'search_results':
        show_search_results()
    elif st.session_state.view_mode == 'detail':
        show_detail()


if __name__ == "__main__":
    main()
