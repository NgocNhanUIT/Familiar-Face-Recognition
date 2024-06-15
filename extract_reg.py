# !git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import faiss
import numpy as np
import os

mtcnn = MTCNN(image_size=160)

resnet = InceptionResnetV1(pretrained='vggface2').eval()


def get_embedding(img_path):
  img = Image.open(img_path)
  img_cropped = mtcnn(img)

  img_embedding = resnet(img_cropped.unsqueeze(0))
  return img_embedding, img_cropped


def save_img(img_cropped, save_path):
    tensor_normalized = (img_cropped + 1) / 2
    tensor_normalized = tensor_normalized.permute(1, 2, 0)* 255

    # Chuyển tensor thành numpy array và kiểu dữ liệu uint8
    numpy_array = tensor_normalized.byte().numpy()

    image = Image.fromarray(numpy_array)
    image.save(save_path)
    

def add_face_db(img_path, face_db_path):
    img_embedding, _ = get_embedding(img_path)

    # Kiểm tra xem file face_db.npy đã tồn tại chưa
    if os.path.exists(face_db_path):
        # Nếu tồn tại, tải dữ liệu hiện có
        face_db = np.load(face_db_path)
        # Thêm embedding mới vào dữ liệu hiện có
        face_db = np.vstack([face_db, img_embedding])
    else:
        # Nếu không tồn tại, tạo một mảng mới với embedding đầu tiên
        face_db = np.array([img_embedding])

    # Lưu dữ liệu đã cập nhật vào file face_db.npy
    np.save(face_db_path, face_db)
    
    
def add_folder_to_face_db(folder_path, face_db_path='face_db.npy'):
    # Duyệt qua tất cả các tệp trong thư mục
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        
        # Kiểm tra xem tệp có phải là ảnh không (giả sử các ảnh có đuôi .jpg, .jpeg, .png)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Adding {img_path} to database")
            add_face_db(img_path, face_db_path)


def load_db(db_path='face_db.npy'):
    """
    Hàm để tải cơ sở dữ liệu từ file .npy
    """
    if os.path.exists(db_path):
        face_db = np.load(db_path)
        return face_db
    else:
        print(f"Database file {db_path} does not exist.")
        return None
    
    
def create_faiss_index(vectors_db):
    # Khởi tạo index faiss.
    index = faiss.IndexFlatL2(vectors_db.shape[1])
    # Thêm các vector vào index.
    index.add(vectors_db)  
    return index


def find_k_nearest_neighbors(query_vector, vectors_db, k):
  # Tính khoảng cách giữa query_vector và các vector trong index.
  distances, indices = vectors_db.search(query_vector.reshape(1, -1), k)
  return indices[0], distances[0]