import sys
import json
import base64
import os
import os.path as osp

import cv2 #pip install opencv-python
import natsort #pip install natsort
import numpy as np #pip install numpy
from shapely.geometry import Polygon #pip install shapely
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox #pip install PyQt5

class LabelTransferApp(QWidget):
    def __init__(self):
        super().__init__()

        # 建立介面
        self.init_ui()

    def init_ui(self):
        # 設定介面標題
        self.setWindowTitle("Label Transfer")

        # 設定視窗大小
        self.resize(600, 300)  # 調整視窗大小至 600x300

        # 垂直佈局
        main_layout = QVBoxLayout()

        # 水平佈局：放置兩個選取資料夾的垂直區塊
        button_layout = QHBoxLayout()

        # 建立資料夾1的區塊（按鈕 + 標籤）
        folder1_layout = QVBoxLayout()
        self.folder1_btn = QPushButton("Source Folder")
        self.folder1_btn.setFixedHeight(60)  # 設定按鈕高度為 60px
        self.folder1_btn.clicked.connect(self.select_folder1)
        self.folder1_label = QLabel("Folder 1: Not selected")
        self.folder1_label.setWordWrap(True)  # 啟用自動換行
        self.folder1_label.setFixedWidth(250)  # 設定標籤寬度限制
        self.folder1_image = QLabel("Number of pictures: ")
        self.folder1_json = QLabel("Number of jsons: ")
        folder1_layout.addWidget(self.folder1_btn)
        folder1_layout.addWidget(self.folder1_label)
        folder1_layout.addWidget(self.folder1_image)
        folder1_layout.addWidget(self.folder1_json)

        # 建立資料夾2的區塊（按鈕 + 標籤）
        folder2_layout = QVBoxLayout()
        self.folder2_btn = QPushButton("Destination Folder")
        self.folder2_btn.setFixedHeight(60)  # 設定按鈕高度為 60px
        self.folder2_btn.clicked.connect(self.select_folder2)
        self.folder2_label = QLabel("Folder 2: Not selected")
        self.folder2_label.setWordWrap(True)  # 啟用自動換行
        self.folder2_label.setFixedWidth(250)  # 設定標籤寬度限制
        self.folder2_image = QLabel("Number of pictures: ")
        self.folder2_json = QLabel("Number of jsons: ")
        folder2_layout.addWidget(self.folder2_btn)
        folder2_layout.addWidget(self.folder2_label)
        folder2_layout.addWidget(self.folder2_image)
        folder2_layout.addWidget(self.folder2_json)

        # 將兩個資料夾的垂直區塊加入水平佈局
        button_layout.addLayout(folder1_layout)
        button_layout.addLayout(folder2_layout)

        # 建立 transfer 按鈕
        self.transfer_btn = QPushButton("Transfer")
        self.transfer_btn.setFixedHeight(60)  # 設定按鈕高度為 60px
        self.transfer_btn.setEnabled(False)  # 預設不可點選
        self.transfer_btn.clicked.connect(self.transfer_annotations)

        # 將水平佈局和 transfer 按鈕加入主佈局
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.transfer_btn)

        # 設定主佈局
        self.setLayout(main_layout)

        # 儲存資料夾路徑
        self.folder1_path = ""
        self.folder2_path = ""

    def select_folder1(self):
        # 選取資料夾1
        folder1 = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if folder1:
            self.folder1_path = folder1
            self.folder1_label.setText(f"Folder 1: {folder1}")
            self.old_image_names, self.old_image_paths, self.old_quantity = self.get_image_from_dir(self.folder1_path)

            self.folder1_image.setText(f"Number of pictures: {self.old_quantity}")
            self.folder1_json.setText(f"Number of jsons: {self.count_json_files(self.folder1_path)}")
        self.check_folders_selected()

    def select_folder2(self):
        # 選取資料夾2
        folder2 = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder2:
            self.folder2_path = folder2
            self.folder2_label.setText(f"Folder 2: {folder2}")
            self.new_image_names, self.new_image_paths, self.new_quantity = self.get_image_from_dir(self.folder2_path)
            self.folder2_image.setText(f"Number of pictures: {self.new_quantity}")
            self.folder2_json.setText(f"Number of jsons: {self.count_json_files(self.folder2_path)}")
        self.check_folders_selected()

    def check_folders_selected(self):
        # 檢查兩個資料夾是否都選取了
        if self.folder1_path and self.folder2_path:
            self.transfer_btn.setEnabled(True)

    def get_image_from_dir(self, path):
        image_names = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
        image_paths = [os.path.join(path, f).replace("\\", "/") for f in image_names]
        image_count = len(image_names)
        return natsort.natsorted(image_names), natsort.natsorted(image_paths), image_count

    def load_image(self, image_path):
        """載入圖片並返回圖片和其尺寸"""
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        return img, width, height

    def read_json(self, image_path):
        """讀取json檔案並返回標記資訊"""
        json_path = os.path.splitext(image_path)[0] + ".json"
        with open(json_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    def count_json_files(self, folder_path):
        # 計算資料夾中的 JSON 檔案數量
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        return len(json_files)

    def rescale_points(self, points, old_width, old_height, new_width, new_height):
        """根據圖片尺寸，重新縮放所有點"""
        scaled_points = []
        for point in points:
            x, y = point
            new_x = (x - self.old_point1[0]) * new_width / old_width + self.new_point1[0]
            new_y = (y - self.old_point1[1]) * new_height / old_height + self.new_point1[1]
            scaled_points.append([new_x, new_y])
        return scaled_points

    # 將圖片數據轉換為base64
    def image_to_base64(self, path):
        with open(path, "rb") as image_file:
            # 將圖片數據編碼為 Base64
            encoded_string = base64.b64encode(image_file.read())
            # 將 bytes 類型的數據轉換為字串
            return encoded_string.decode('utf-8')

    # 檢查多邊形是否為順時針，否則反轉方向
    def ensure_clockwise(self, polygon):
        poly = Polygon(polygon)
        if poly.exterior.is_ccw:  # 如果是逆時針，反轉頂點順序
            return np.flipud(polygon)
        return polygon

    # 將多邊形起始點與另一個對齊，使用最近點作為起始
    def align_starting_point(self, poly1, poly2):
        min_dist = float('inf')
        best_idx = 0
        for i in range(len(poly2)):
            dist = np.linalg.norm(poly1[0] - poly2[i])
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        # 重新排列poly2，使其與poly1的起始點對齊
        poly2 = np.roll(poly2, -best_idx, axis=0)
        return poly2

    # 計算多邊形頂點數量並重新取樣
    def resample_polygon(self, polygon, num_points):
        poly = Polygon(polygon)  # 將多邊形頂點創建為 Polygon
        exterior = poly.exterior  # 取出多邊形的邊界（外部環）
        
        # 使用線性距離重新取樣多邊形的邊界點
        distances = np.linspace(0, exterior.length, num_points)
        
        return np.array([exterior.interpolate(distance).coords[0] for distance in distances])


    # 內插兩個多邊形的頂點，得到中間時期的頂點
    def interpolate_polygons(self, poly1, poly2, num_points, index, start, end):
        poly1 = self.ensure_clockwise(poly1)
        poly2 = self.ensure_clockwise(poly2)

        if len(poly1) != num_points:
            poly1 = self.resample_polygon(poly1, num_points)
        if len(poly2) != num_points:
            poly2 = self.resample_polygon(poly2, num_points)

        poly2 = self.align_starting_point(poly1, poly2)  # 對齊起始點
        d = end - start
        return (poly1 * (end - index) / d + poly2 * (index - start) / d)

    # 將內插結果保存為JSON文件
    def save(self, data, file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)

    def finish_msg(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Labels transferred successfully")
        msg.setText("Labels successfully transferred to the new images!")
        msg.setStandardButtons(QMessageBox.Ok)

        msg.exec_()

    def transfer_annotations(self):
        # 列出資料夾中的所有檔案
        self.files_in_folder = os.listdir(self.folder2_path)
        self.news = []

        self.old_point1 = (62, 132)  # ADC左上角頂點，座標(X,Y)
        self.old_point2 = (312, 325)  # ADC右下角頂點
        self.new_point1 = (0, 95)      # DCE左上角頂點
        self.new_point2 = (352, 325)    # DCE右下角頂點

        old_width = self.old_point2[0] - self.old_point1[0]
        old_height = self.old_point2[1] - self.old_point1[1]
        new_width = self.new_point2[0] - self.new_point1[0]
        new_height = self.new_point2[1] - self.new_point1[1]

        # 根據數量較少的圖片數量當作重複次數
        copy_times = self.old_quantity if self.old_quantity <= self.new_quantity else self.new_quantity
        # 根據圖片之間的空缺數量決定index變量
        if self.old_quantity != 1 and self.new_quantity != 1:
            var = (self.new_quantity-1) / (self.old_quantity - 1) if self.old_quantity <= self.new_quantity else (self.old_quantity-1) / (self.new_quantity - 1)
        else:
            var = self.new_quantity if self.old_quantity <= self.new_quantity else self.old_quantity

        # 第一部分，將現有的標記轉移至對應的圖片
        for number in range(copy_times): 
            old_id = number if self.old_quantity <= self.new_quantity else int(var * number + 0.5)
            new_id = int(var * number + 0.5) if self.old_quantity <= self.new_quantity else number
            # 載入舊圖片與新圖片，取得圖片尺寸
            # old_img, old_width, old_height = self.load_image(self.old_image_paths[old_id])
            # new_img, new_width, new_height = self.load_image(self.new_image_paths[new_id])

            # 讀取標記資訊，沒有標記的話則跳過
            old_json_name = os.path.splitext(self.old_image_names[old_id])[0] + ".json"
            if old_json_name not in os.listdir(self.folder1_path):
                continue
            annotations = self.read_json(self.old_image_paths[old_id])
            
            # 遍歷每個標記，將其縮放並轉換為新圖片的對應座標
            new_shapes = []
            for shape in annotations['shapes']:
                old_points = shape['points']
                new_points = self.rescale_points(old_points, old_width, old_height, new_width, new_height)
                new_shapes.append({
                    'label': shape['label'],
                    'points': new_points,
                    'group_id': shape['group_id'],
                    'description': shape['description'],
                    'shape_type': shape['shape_type'],
                    'flags': shape['flags'],
                    'mask': shape['mask']
                })
            output_json_path = os.path.splitext(self.new_image_paths[new_id])[0] + ".json" #含父資料夾
            output_json_name = os.path.splitext(self.new_image_names[new_id])[0] + ".json" #純資料名稱

            # 如果已存在JSON檔案，則讀取內容並新增
            if output_json_name in self.files_in_folder:
                original = self.read_json(self.new_image_paths[new_id])
                new_shapes += [
                    dict(
                        label=s["label"],
                        points=s["points"],
                        shape_type=s.get("shape_type", "polygon"),
                        flags=s.get("flags", {}),
                        description=s.get("description"),
                        group_id=s.get("group_id"),
                        mask=s["mask"]
                        if s.get("mask")
                        else None,
                        other_data=s.get("other_data")
                    )
                    for s in original["shapes"]
                ]
                original['shapes'] = new_shapes
                self.save(original, output_json_path)
            else:
                self.files_in_folder += [output_json_name]
                data = dict(
                        version=annotations['version'],
                        flags=annotations['flags'],
                        shapes=new_shapes,
                        imagePath=self.new_image_paths[new_id].replace("\\", "/"),
                        imageData=self.image_to_base64(self.new_image_paths[new_id]),
                        imageHeight=new_height,
                        imageWidth=new_width,
                    )
                self.save(data, output_json_path)
            self.news.append(new_id)

        # 第二部分，將中間空餘的圖片透過內插法去預測標記
        for number in range(copy_times - 1):
            new_id_start = int(var * number + 0.5)
            new_id_end = int(var * (number + 1) + 0.5)
            if(new_id_start not in self.news or new_id_end not in self.news):continue
            for index in range(new_id_start + 1, new_id_end):
                # 讀取前面移轉的數據
                new_json_name1 = os.path.splitext(self.new_image_names[new_id_start])[0] + ".json"
                new_json_name2 = os.path.splitext(self.new_image_names[new_id_end])[0] + ".json"
                if (new_json_name1 not in self.files_in_folder) or (new_json_name2 not in self.files_in_folder):
                    continue
                data1 = self.read_json(self.new_image_paths[new_id_start])
                data2 = self.read_json(self.new_image_paths[new_id_end])

                # 讀取數據中的標記，**目前功能只允許內插第一個標記**
                poly1 = np.array(data1['shapes'][0]['points'])
                poly2 = np.array(data2['shapes'][0]['points'])

                # 重新取樣並內插多邊形
                num_points = max(len(poly1), len(poly2))
                interpolated_poly = self.interpolate_polygons(poly1, poly2, num_points, index, new_id_start, new_id_end)

                # 保存結果到新JSON文件
                output_json_path = os.path.splitext(self.new_image_paths[index])[0] + ".json" #含父資料夾
                output_json_name = os.path.splitext(self.new_image_names[index])[0] + ".json" #純資料名稱
                if output_json_name in self.files_in_folder:
                    original = self.read_json(self.new_image_paths[index])
                    new_shapes = []
                    new_shapes.append({
                        'label': "label_inserted",
                        'points': interpolated_poly.tolist(),
                        'group_id': None,
                        'description': "",
                        'shape_type': "polygon",
                        'flags': {},
                        'mask': None
                    })
                    new_shapes += [
                        dict(
                            label=s["label"],
                            points=s["points"],
                            shape_type=s.get("shape_type", "polygon"),
                            flags=s.get("flags", {}),
                            description=s.get("description"),
                            group_id=s.get("group_id"),
                            mask=s["mask"]
                            if s.get("mask")
                            else None,
                            other_data=s.get("other_data")
                        )
                        for s in original["shapes"]
                    ]
                    original['shapes'] = new_shapes
                    self.save(original, output_json_path)
                else:
                    new_img, new_width, new_height = self.load_image(self.new_image_paths[index])
                    data = dict(
                            version=data1['version'],
                            flags=data1['flags'],
                            shapes=data1['shapes'],
                            imagePath=self.new_image_paths[index].replace("\\", "/"),
                            imageData=self.image_to_base64(self.new_image_paths[index]),
                            imageHeight=new_height,
                            imageWidth=new_width,
                        )
                    data['shapes'][0]['label'] = "label_inserted"
                    data['shapes'][0]['points'] = interpolated_poly.tolist()  # 更新多邊形頂點
                    self.save(data, output_json_path)
        
        self.finish_msg()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelTransferApp()
    window.show()
    sys.exit(app.exec_())
