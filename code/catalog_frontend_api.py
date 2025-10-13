import streamlit as st
import requests
from PIL import Image
import io
import random
from pathlib import Path
import base64

# API endpoint
API_URL = "http://localhost:8000"

# Base image directory
BASE_IMAGE_DIR = r"C:\Users\User\OneDrive\Desktop\tinyvlm\data\image-description-marketplace-data\flip_data_vlm\flip_data_vlm"

# Initialize session state
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'catalog'  # 'catalog', 'search_results', 'detail'
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'random_images' not in st.session_state:
    st.session_state.random_images = []


def get_all_images():
    """Get all image paths from the base directory."""
    image_paths = []
    base_path = Path(BASE_IMAGE_DIR)
    
    # Search in category folders and flo_images
    for pattern in ['category_*/images/*.jpg', 'flo_images/*.jpg']:
        image_paths.extend(list(base_path.glob(pattern)))
    
    return [str(p) for p in image_paths]


def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def load_random_images(count=50):
    """Load random images for the catalog."""
    all_images = get_all_images()
    if len(all_images) > count:
        return random.sample(all_images, count)
    return all_images


def search_by_text(text_query):
    """Search images by text query."""
    try:
        response = requests.post(
            f"{API_URL}/encode_text_query",
            params={"text_query": text_query},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['top_k_similar_items']
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
        return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []


def search_by_image(image_file):
    """Search images by uploading an image."""
    try:
        image_file.seek(0)  # Reset file pointer
        files = {"image_query": ("image.jpg", image_file, "image/jpeg")}
        response = requests.post(f"{API_URL}/encode_image_query", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()['top_k_similar_items']
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
        return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Image search failed: {e}")
        return []


def get_similar_images(image_path):
    """Get similar images for a given image."""
    try:
        with open(image_path, 'rb') as f:
            files = {"image_query": ("image.jpg", f, "image/jpeg")}
            response = requests.post(f"{API_URL}/encode_image_query", files=files, timeout=30)
            if response.status_code == 200:
                return response.json()['top_k_similar_items']
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        return []
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI server is running on http://localhost:8000")
        return []
    except Exception as e:
        st.error(f"Failed to get similar images: {e}")
        return []


def display_image_grid(image_list, cols=3):
    """Display images in a grid layout."""
    rows = [image_list[i:i+cols] for i in range(0, len(image_list), cols)]
    
    for row in rows:
        columns = st.columns(cols)
        for idx, col in enumerate(columns):
            if idx < len(row):
                image_path = row[idx] if isinstance(row[idx], str) else row[idx][0]
                try:
                    with col:
                        # Check if file exists
                        if not Path(image_path).exists():
                            col.error(f"Image not found")
                            continue
                            
                        img = Image.open(image_path)
                        # Don't resize - keep original resolution or resize larger
                        # Calculate size to maintain aspect ratio
                        max_size = 1200
                        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        
                        # Display image
                        st.image(img, use_column_width=True)
                        
                        # Make image clickable with a simple button
                        if st.button("Click to view details", key=f"btn_{image_path}", use_container_width=True):
                            st.session_state.selected_image = image_path
                            st.session_state.view_mode = 'detail'
                            st.rerun()
                        
                        # Show filename
                        st.caption(Path(image_path).name)
                        # Show score if available
                        if isinstance(row[idx], tuple):
                            st.caption(f"Score: {row[idx][1]:.3f}")
                except Exception as e:
                    col.error(f"Error: {str(e)}")


def show_catalog():
    """Show the main catalog page."""
    st.title("🖼️ Image Catalog & Search")
    
    # API Status check
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        if response.status_code == 200:
            st.success("✅ API Connected")
        else:
            st.warning("⚠️ API responding but status unclear")
    except:
        st.error("❌ API Not Connected - Start FastAPI server first!")
    
    # Search section
    st.subheader("Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        text_query = st.text_input("🔍 Search by text", placeholder="Enter description...")
        if st.button("Search by Text", use_container_width=True):
            if text_query:
                with st.spinner("Searching..."):
                    results = search_by_text(text_query)
                    if results:
                        st.session_state.search_results = results[:10]
                        st.session_state.view_mode = 'search_results'
                        st.rerun()
                    else:
                        st.warning("No results found or API error")
    
    with col2:
        uploaded_file = st.file_uploader("📸 Search by image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            with st.spinner("Searching..."):
                results = search_by_image(uploaded_file)
                if results:
                    st.session_state.search_results = results[:10]
                    st.session_state.view_mode = 'search_results'
                    st.rerun()
                else:
                    st.warning("No results found or API error")
    
    st.divider()
    
    # Random catalog
    st.subheader("Browse Catalog")
    
    if st.button("🔄 Refresh Catalog"):
        st.session_state.random_images = load_random_images(50)
    
    if not st.session_state.random_images:
        with st.spinner("Loading images..."):
            st.session_state.random_images = load_random_images(50)
    
    if st.session_state.random_images:
        display_image_grid(st.session_state.random_images, cols=3)
    else:
        st.warning("No images found in the directory. Check BASE_IMAGE_DIR path.")


def show_search_results():
    """Show search results page."""
    st.title("🔍 Search Results")
    
    if st.button("← Back to Catalog"):
        st.session_state.view_mode = 'catalog'
        st.rerun()
    
    st.divider()
    
    if st.session_state.search_results:
        st.subheader(f"Top {len(st.session_state.search_results)} Results")
        display_image_grid(st.session_state.search_results, cols=3)
    else:
        st.info("No results found.")


def show_detail():
    """Show detailed view of selected image with similar items."""
    st.title("📸 Image Details")
    
    if st.button("← Back"):
        if st.session_state.get('previous_view') == 'search_results':
            st.session_state.view_mode = 'search_results'
        else:
            st.session_state.view_mode = 'catalog'
        st.rerun()
    
    st.divider()
    
    if st.session_state.selected_image:
        try:
            # Check if file exists
            if not Path(st.session_state.selected_image).exists():
                st.error(f"Image not found: {st.session_state.selected_image}")
                return
                
            # Display selected image
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                img = Image.open(st.session_state.selected_image)
                st.image(img, use_column_width=True)
                st.subheader(Path(st.session_state.selected_image).name)
            
            st.divider()
            
            # Get and display similar images
            st.subheader("🔗 Similar Items")
            
            with st.spinner("Finding similar images..."):
                similar_items = get_similar_images(st.session_state.selected_image)
            
            if similar_items:
                # Exclude the selected image itself if it appears in results
                similar_items = [item for item in similar_items 
                               if item[0] != st.session_state.selected_image][:5]
                display_image_grid(similar_items, cols=3)
            else:
                st.info("No similar images found.")
                
        except Exception as e:
            st.error(f"Failed to load image: {e}")
    else:
        st.warning("No image selected.")


# Main app logic
def main():
    st.set_page_config(
        page_title="Image Search & Catalog",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
        }
        div[data-testid="stImage"] {
            transition: transform 0.2s;
        }
        div[data-testid="stImage"]:hover {
            transform: scale(1.02);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Route to appropriate view
    if st.session_state.view_mode == 'catalog':
        show_catalog()
    elif st.session_state.view_mode == 'search_results':
        show_search_results()
    elif st.session_state.view_mode == 'detail':
        show_detail()


if __name__ == "__main__":
    main()