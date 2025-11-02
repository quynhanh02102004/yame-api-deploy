# CÁC THƯ VIỆN CẦN THIẾT
import os
import gdown
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import faiss
from PIL import Image
import numpy as np
# import pandas as pd # Không thấy dùng pandas trong code gốc, có thể bỏ
import io
import torchvision.transforms as transforms

# --- PHẦN MỚI: TỰ ĐỘNG TẢI MODEL TỪ GOOGLE DRIVE ---
# Code này sẽ tạo các thư mục cần thiết và tải file từ Google Drive
# nếu chúng chưa tồn tại trên server.

# === FILE IDs ĐÃ ĐƯỢC ĐIỀN SẴN DỰA TRÊN LINK BẠN CUNG CẤP ===
file_configs = {
    "models/model.pth": "1eUhNvt3r6I1oPJ58fDjH6LLBG4UdeiaJ",
    "data/similarity/faiss_index.index": "1RVy1JhdHziVYujskKuhNbClfOj9X0B9J",
    "data/similarity/vectors.npy": "1Ap_ANmjiEeJteDiK_PmagXT-zy8sZDbn",
    "data/similarity/id_map.json": "1jpyA03eK6_U5YWRlFHkQulYAv5e95_wn",
    "data/items_metadata_joined_fixed.json": "1sLzSQJwE5PAanEWCJbQPowaJvhC7LOQR"
}
# =============================================================

# Hàm để tải file, tự động tạo thư mục cha nếu cần
def download_files():
    print("Checking for model and data files...")
    for relative_path, file_id in file_configs.items():
        # Tạo thư mục cha nếu nó không tồn tại
        dir_name = os.path.dirname(relative_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            
        if not os.path.exists(relative_path):
            print(f"Downloading {relative_path}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, relative_path, quiet=False)
            print(f"{relative_path} downloaded successfully.")
        else:
            print(f"{relative_path} already exists. Skipping download.")

# Chạy hàm tải file ngay khi ứng dụng khởi động
download_files()
# ----------------------------------------------------


# KHỞI TẠO ỨNG DỤNG FASTAPI
app = FastAPI(title='Recommendation System API')

# CẤU HÌNH CORS
origins = [
    "http://localhost:3000",
    "https://yame-clone-animated.vercel.app"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- TẢI MODEL VÀ DỮ LIỆU TỪ CÁC FILE ĐÃ TẢI VỀ ---
print("Loading models and data from local files...")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Tải model ResNet
model = torch.load('models/model.pth', map_location=device)
model.eval()
print("PyTorch model loaded successfully.")

# Tải dữ liệu cho Similarity
similarity_vectors = np.load('data/similarity/vectors.npy')
with open('data/similarity/id_map.json', 'r') as f:
    similarity_id_map = json.load(f)
similarity_faiss_index = faiss.read_index('data/similarity/faiss_index.index')
print("Similarity data loaded.")

# Tải metadata sản phẩm
with open('data/items_metadata_joined_fixed.json', 'r') as f:
    items_metadata = json.load(f)
print("Items metadata loaded.")

# Định nghĩa các bước chuyển đổi hình ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("Image transformer created.")
print("--- Server is ready to accept requests ---")
# ------------------------------------------------

# --- CÁC API ENDPOINTS (Lấy từ repo của bạn) ---
@app.get("/")
def read_root():
    return {"message": "Welcome to Recommendation API!"}

@app.post("/api/recommend_by_image")
async def recommend_by_image(file: UploadFile = File(...)):
    """
    Nhận một file ảnh và trả về các sản phẩm tương tự.
    """
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Trích xuất vector từ ảnh
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_vector = model(image_tensor).cpu().numpy()

    # Tìm kiếm trong Faiss
    k = 10  # Số lượng sản phẩm gợi ý
    distances, indices = similarity_faiss_index.search(query_vector, k)
    
    # Lấy ID sản phẩm và thông tin metadata
    results = []
    for i in indices[0]:
        item_id = similarity_id_map.get(str(i)) # Dùng .get() để an toàn hơn
        if item_id:
            item_info = items_metadata.get(item_id, {})
            if item_info:
                results.append({
                    "id": item_id,
                    "name": item_info.get("name"),
                    "image": item_info.get("image"),
                    "price": item_info.get("price"),
                    "brand": item_info.get("brand") # Thêm brand nếu có
                })
            
    return {"results": results}