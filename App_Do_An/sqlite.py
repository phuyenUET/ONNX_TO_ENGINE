import sqlite3
from datetime import datetime
import csv
import os

class BottleDB:
    @staticmethod
    def init_db():
        """Tạo bảng nếu chưa tồn tại"""
        with sqlite3.connect("bottle_log.db") as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS bottles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bottle_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_type TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            conn.commit()

    @staticmethod
    def insert_data(bottle_id, status, error_type):
        """Chèn dữ liệu chai mới vào database"""
        with sqlite3.connect("bottle_log.db") as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO bottles (bottle_id, status, error_type, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (
                bottle_id,
                status,
                error_type,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()

    @staticmethod
    def export_to_csv(filename="bottle_log.csv"):
        """Xuất toàn bộ dữ liệu ra file CSV"""
        with sqlite3.connect("bottle_log.db") as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM bottles")
            rows = c.fetchall()
            headers = [desc[0] for desc in c.description]

        with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"✅ Đã xuất ra: {os.path.abspath(filename)}")


# # Test
# if __name__ == "__main__":
#     BottleDB.init_db()
#     BottleDB.insert_data("1", "OK", "0")
#     BottleDB.insert_data("2", "ERROR", "1")
#     BottleDB.export_to_csv()

