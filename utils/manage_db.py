import os
import numpy as np
import pandas as pd
from PIL import Image
import faiss
import sqlite3
import datetime

import sys
sys.path.append(r"./")
from utils.models import ModelManager

class ManageDB(ModelManager):
    def __init__(self, face_db_path=r'./data/face_db.csv'):
        super().__init__()
        self.face_db_path = face_db_path
        self.face_db = self.load_db()
        self.index_faiss = self.create_faiss_index()


    def get_embedding(self, img_path):
        img = Image.open(img_path)
        img_cropped = self.mtcnn(img)

        if img_cropped is not None:
            # đầu ra vector đã chuẩn hóa
            img_embedding = self.resnet(img_cropped.unsqueeze(0))
            return img_embedding.detach().numpy().flatten(), img_cropped
        else:
            return None, None

    def save_img(self, img_cropped, save_path):
        tensor_normalized = (img_cropped + 1) / 2
        tensor_normalized = tensor_normalized.permute(1, 2, 0) * 255

        numpy_array = tensor_normalized.byte().numpy()
        image = Image.fromarray(numpy_array)
        image.save(save_path)
        
    def add_face_db(self, img_path, name = None):
        img_embedding, _ = self.get_embedding(img_path)
        if img_embedding is not None:
            if name is None:
                name = os.path.basename(img_path).split('.')[0][:-5].replace('_', ' ')
            new_entry = pd.DataFrame([[name] + img_embedding.tolist()], columns=['name'] + [f'vector_{i}' for i in range(512)])
            if self.face_db is not None:
                self.face_db = pd.concat([self.face_db, new_entry], ignore_index=True)
            else:
                self.face_db = new_entry

            # Ghi vào file CSV
            self.face_db.to_csv(self.face_db_path, index=False)
            self.index_faiss = self.create_faiss_index()
            print("add successfully")

    def add_folder_to_face_db(self, folder_path):
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Adding {img_path} to database")
                self.add_face_db(img_path)

    def load_db(self):
        if os.path.exists(self.face_db_path):
            face_db = pd.read_csv(self.face_db_path)
            return face_db
        else:
            print(f"Database file {self.face_db_path} does not exist. Creating a new one.")
            # Tạo một DataFrame trống với các cột thích hợp
            columns = ['name'] + [f'vector_{i}' for i in range(512)]
            face_db = pd.DataFrame(columns=columns)
            # Lưu DataFrame trống vào file CSV
            face_db.to_csv(self.face_db_path, index=False)
            return face_db

    def create_faiss_index(self):
        if self.face_db is not None and not self.face_db.empty:
            # Sử dụng IP:Inner Product (tích vô hướng) vì vector đã chuẩn hóa nên tích vô hướng tương đương cosine similarity
            index_faiss = faiss.IndexFlatIP(512)  
            vectors = self.face_db.iloc[:, 1:].values.astype('float32')
            index_faiss.add(vectors)
            return index_faiss
        else:
            print("Face database is empty. Cannot create FAISS index.")
            return None

    def find_k_nearest_neighbors(self, query_vector, k, threshold = 0.5):
        if self.index_faiss is not None:
            query_vector = query_vector / np.linalg.norm(query_vector)  # Chuẩn hóa vector truy vấn
            distances, indices = self.index_faiss.search(query_vector.reshape(1, -1), k)
            
            # Lọc các kết quả dựa trên ngưỡng
            valid_indices = np.where(distances[0] >= threshold)[0]
            filtered_indices = indices[0][valid_indices]
            filtered_distances = distances[0][valid_indices]
            list_name = self.face_db.iloc[filtered_indices, 0].tolist()
            
            return filtered_indices, filtered_distances, list_name
        else:
            print("FAISS index has not been created.")
            return None, None, None
    
    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("./data/attendance.db")
        cursor = conn.cursor()

        # Create the 'attendance' table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                date TEXT,
                time TEXT
            )
        """)
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        # existing_entry = cursor.fetchone()

        # if existing_entry:
        #     # print(f"{name} is already marked as present for {current_date}")
        #     pass
        # else:
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
        conn.commit()
        # print(f"{name} marked as present for {current_date} at {current_time}")

        conn.close()

if __name__ == "__main__":
    file_path = r"./data/face_db.csv"
    manager = ManageDB(face_db_path=file_path)
    manager.load_recognition_model()
    
    # manager.add_folder_to_face_db(r"C:\Users\admin\Downloads\reg\lfw\Charles_Bronson")
    # manager.add_folder_to_face_db(r"C:\Users\admin\Downloads\reg\lfw\Emma_Thompson")
    manager.add_face_db(r"C:\Users\admin\Pictures\Camera Roll\img1.jpg", "thanhstar")
    
    # img_embedding, img_cropped = manager.get_embedding(r"C:\Users\admin\Downloads\reg\lfw\Charles_Bronson\Charles_Bronson_0003.jpg")
    # manager.save_img(img_cropped, "face_cropped.jpg")
    # find_indices, find_distances, names = manager.find_k_nearest_neighbors(img_embedding, k=3, threshold=0.3)
    
    # print(find_indices)
    # print(find_distances)
    # print(names)