import os
import numpy as np
import pandas as pd
from PIL import Image
import faiss
import sqlite3
import datetime
from pinecone import Pinecone
import os

import sys
sys.path.append(r"./")

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')

from utils.models import ModelManager

class ManageDB(ModelManager):
    def __init__(self):
        super().__init__()
        self.pinecone_index = self.create_pinecone_index()
        self.len_db = self.len_pinecone()
        self.attendance_db_path = "./data/attendance.db"

    def len_pinecone(self):
        index_stats = self.pinecone_index.describe_index_stats()
        num_vectors = index_stats['total_vector_count']
        return num_vectors
    
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
        vectors = []
        if img_embedding is not None:
            if name is None:
                name = os.path.basename(img_path).split('.')[0][:-5].replace('_', ' ')
            vectors.append({
                                'id': f"{name}{self.len_db}",
                                'values': img_embedding.tolist(),
                                'metadata': {'name': name}
                            })
            self.pinecone_index.upsert(vectors)
            self.len_db += 1
            print("add successfully")
        else:
            print("add failed")

    def create_pinecone_index(self):
        pc = Pinecone(api_key=api_key)
        # Kết nối tới index
        pinecone_index = pc.Index('face-reg')
        return pinecone_index
    
    
    def retrieval_pinecone(self, query_vector, k=3,threshold = 0.5):
        query_vector = query_vector.tolist()
        query_results = self.pinecone_index.query(
                            vector=query_vector,
                            top_k=k,
                            # include_values=True
                            include_metadata=True
                        )
        name = query_results["matches"][0]['metadata']["name"]
        score = query_results["matches"][0]['score']
        print(name, score)
        
        if score > threshold:
            return name, score
        else:
            return "Unknown", 0
        
    
    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect(self.attendance_db_path)
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
    
    # manager.add_face_db(r"C:\Users\admin\Pictures\Camera Roll\img1.jpg", "thanhstar")
    
    img_path = r"C:\Users\admin\Pictures\Camera Roll\img1.jpg"
    img_embedding, img_cropped = manager.get_embedding(img_path)
    name,score = manager.retrieval_pinecone(img_embedding, k=3, threshold=0.8)