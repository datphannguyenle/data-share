import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
import mediapipe as mp
from ultralytics import YOLO

class RealSensePointCloud:
    def __init__(self, width=640, height=480, fps=30):
        """
        Khởi tạo camera RealSense với chức năng Point Cloud
        """
        # Khởi tạo pipeline
        self.pipeline = rs.pipeline()
        
        # Cấu hình camera
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Khởi động camera
        self.pipeline.start(self.config)
        
        # Căn chỉnh depth với color
        self.align = rs.align(rs.stream.color)
        self.pc = rs.pointcloud()
        
        # Bộ lọc depth
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.model = YOLO("D:\\yolov11\\best.pt")
        
        # Load class names
        with open("D:\\yolov11\\classes.txt", "r") as f:
            self.class_names = f.read().splitlines()

    def get_frames(self):
        """
        Lấy frame màu và depth từ camera, áp dụng căn chỉnh và lọc
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        depth_frame = self.decimation_filter.process(depth_frame)
        depth_frame = self.spatial_filter.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)

        return color_frame, depth_frame
    
    def detect_human(self, color_image):
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                x, y = int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0])
                cv2.circle(mask, (x, y), 50, 255, -1)
        return mask
    
    def detect_pig(self, color_image):
        results = self.model(color_image)[0]
        mask = np.zeros(color_image.shape[:2], dtype=np.uint8)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if self.model.names[cls_id] == 'pig':  # hoặc kiểm tra theo id nếu bạn biết ID lớp heo
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    def create_point_cloud(self):
        """
    Tạo Point Cloud và lọc chỉ giữ lại phần của người.
    """
        color_frame, depth_frame = self.get_frames()
        if not color_frame or not depth_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())

    # Lấy mask nhận diện người, heo
        mask = self.detect_pig(color_image)

        self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        depth_video_frame = depth_frame.as_video_frame()
        depth_width = depth_video_frame.get_width()
        depth_height = depth_video_frame.get_height()

    # Lấy tọa độ 3D
        vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

    # 🔥 Resize mask để khớp với Point Cloud
        mask_resized = cv2.resize(mask, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)

        mask_flat = mask_resized.flatten() > 0

        if len(mask_flat) != len(vtx):
            print(f"⚠️ Kích thước không khớp: mask={len(mask_flat)}, vtx={len(vtx)}")
            return None

    # Áp dụng mask để lọc chỉ lấy điểm của người
        vtx = vtx[mask_flat]

    # ✅ Resize color_image trước khi reshape
        color_resized = cv2.resize(color_image, (depth_width, depth_height), interpolation=cv2.INTER_NEAREST)
        tex = color_resized.reshape(-1, 3) / 255.0  # Đưa về range [0,1]
        tex = np.clip(tex, 0, 1)
        tex = tex[mask_flat]  # Lọc màu cho đúng điểm 3D

        if vtx.shape[0] == 0:
            print("Không tìm thấy điểm thuộc người.")
            return None

    # Tạo Point Cloud
        o3d.visualization.webrtc_server.enable_webrtc()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx)
        pcd.colors = o3d.utility.Vector3dVector(tex)

        return pcd

    def visualize_point_cloud(self):
        """
        Hiển thị Point Cloud bằng Open3D
        """
        pcd = self.create_point_cloud()
        if pcd is not None:
            o3d.visualization.draw_geometries([pcd])
        else:
            print("Không thể tạo Point Cloud!")

    def get_pointcloud_at(self, x, y):
        """
        Lấy tọa độ 3D tại vị trí pixel (x, y)
        """
        color_frame, depth_frame = self.get_frames()
        if not color_frame or not depth_frame:
            return None

        # Lấy thông tin về camera intrinsics
        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        # Kiểm tra nếu (x, y) nằm trong phạm vi ảnh
        depth_width = depth_frame.get_width()
        depth_height = depth_frame.get_height()
        if not (0 <= x < depth_width and 0 <= y < depth_height):
            print(f" Vị trí ({x}, {y}) ngoài phạm vi ảnh depth ({depth_width}x{depth_height})")
            return None

        # Lấy giá trị độ sâu tại (x, y)
        depth = depth_frame.get_distance(x, y)

        if depth == 0:
            print(f" Điểm ({x}, {y}) không có dữ liệu!")
            return None

        # Chuyển đổi từ pixel sang tọa độ không gian 3D (x, y, z)
        point_3D = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)

        return np.array(point_3D)

    def interactive_capture(self):
        """
        Chế độ hiển thị và điều khiển bằng bàn phím:
        - 'p': Hiển thị Point Cloud
        - 's': Lưu Point Cloud
        - 'q': Thoát
        """

        def mouse_callback(event, x, y, flags, param):
            """
            Xử lý sự kiện click chuột để lấy tọa độ 3D
            """
            if event == cv2.EVENT_LBUTTONDOWN:
                point_3D = self.get_pointcloud_at(x, y)
                if point_3D is not None:
                    print(f" Tọa độ 3D tại ({x}, {y}): {point_3D}")

        try:
            cv2.namedWindow('RealSense')
            cv2.setMouseCallback('RealSense', mouse_callback)

            while True:
                # Lấy frames
                color_frame, depth_frame = self.get_frames()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )

                # Resize depth_colormap để khớp với color_image
                depth_colormap_resized = cv2.resize(
                    depth_colormap, (color_image.shape[1], color_image.shape[0])
                )

                # Ghép 2 ảnh
                images = np.hstack((color_image, depth_colormap_resized))

                cv2.imshow('RealSense', images)

                key = cv2.waitKey(1)

                if key == ord('p'):
                    self.visualize_point_cloud()
                    pcd = self.create_point_cloud()
                    if pcd is not None:
                        
                        filename = f"realsense_pointcloud_{int(time.time())}.ply"
                        o3d.io.write_point_cloud(filename, pcd)
                        print(f"Point Cloud đã được lưu thành: {filename}")
                elif key == ord('s'):
                    pcd = self.create_point_cloud()
                    if pcd is not None:
                        o3d.io.write_point_cloud("realsense_pointcloud.ply", pcd)
                        print("Point Cloud đã được lưu.")
                elif key == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()

    def __del__(self):
        """
        Giải phóng tài nguyên
        """
        try:
            self.pipeline.stop()
        except:
            pass

def main():
    """
    Chương trình chính
    """
    camera = RealSensePointCloud()
    camera.interactive_capture()

if __name__ == "__main__":
    main()
