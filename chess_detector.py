# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

class ChessDetector:
    def __init__(self, board_size=15, warp_size=750):
        """
        初始化五子棋检测器
        Args:
            board_size (int): 棋盘尺寸，默认为15x15
            warp_size (int): 透视变换后图像的边长（像素）
        """
        self.BOARD_SIZE = board_size
        self.WARP_SIZE = warp_size
        
        # 颜色范围定义 (HSV色彩空间)
        self.LOWER_YELLOW = np.array([9, 126, 195])
        self.UPPER_YELLOW = np.array([29, 227, 255])
        self.LOWER_BLACK = np.array([0, 0, 0])
        self.UPPER_BLACK = np.array([180, 255, 70])
        
        # 霍夫圆检测参数
        self.HOUGH_DP = 1
        self.HOUGH_MIN_DIST = int(warp_size / (board_size * 1.8))
        self.HOUGH_PARAM1 = 100
        self.HOUGH_PARAM2 = 18
        self.HOUGH_MIN_RADIUS = int(warp_size / (board_size * 2 * 1.8))
        self.HOUGH_MAX_RADIUS = int(warp_size / (board_size * 2 * 0.8))
        
        # 棋子颜色判断阈值
        self.PIECE_WHITE_THRESH = 160
        self.PIECE_BLACK_THRESH = 100
        
        # 棋盘线检测参数
        self.LINE_THRESHOLD = 100
        self.LINE_MIN_LENGTH = 50
        self.LINE_MAX_GAP = 20
        self.POINT_DISTANCE_THRESHOLD = 10  # 点之间的最小距离阈值

    def order_points(self, pts):
        """
        对输入的四个点进行排序，顺序为：左上, 右上, 右下, 左下
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def detect_board_lines(self, warped_board):
        """
        检测棋盘线
        Args:
            warped_board: 透视变换后的棋盘图像
        Returns:
            tuple: (水平线列表, 垂直线列表)
        """
        # 转换为灰度图
        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 使用霍夫线变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.LINE_THRESHOLD,
                               minLineLength=self.LINE_MIN_LENGTH,
                               maxLineGap=self.LINE_MAX_GAP)
        
        if lines is None:
            return [], []
        
        # 将直线分为水平线和垂直线
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线的角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 根据角度分类
            if angle < 10 or angle > 170:  # 水平线
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # 垂直线
                vertical_lines.append(line[0])
        
        return horizontal_lines, vertical_lines

    def find_outer_corners(self, warped_board):
        """
        找到棋盘最外圈的交叉点
        Args:
            warped_board: 透视变换后的棋盘图像
        Returns:
            list: 最外圈交叉点列表，按顺序排列
        """
        # 转换为灰度图
        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 使用霍夫线变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.LINE_THRESHOLD,
                               minLineLength=self.LINE_MIN_LENGTH,
                               maxLineGap=self.LINE_MAX_GAP)
        
        if lines is None:
            return []
        
        # 将直线分为水平线和垂直线
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线的角度
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # 根据角度分类
            if angle < 10 or angle > 170:  # 水平线
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # 垂直线
                vertical_lines.append(line[0])
        
        # 找到最外圈的线
        outer_h_lines = []
        outer_v_lines = []
        
        # 按y坐标排序水平线
        horizontal_lines.sort(key=lambda x: (x[1] + x[3])/2)
        # 取最上和最下的线
        if len(horizontal_lines) >= 2:
            outer_h_lines.append(horizontal_lines[0])
            outer_h_lines.append(horizontal_lines[-1])
        
        # 按x坐标排序垂直线
        vertical_lines.sort(key=lambda x: (x[0] + x[2])/2)
        # 取最左和最右的线
        if len(vertical_lines) >= 2:
            outer_v_lines.append(vertical_lines[0])
            outer_v_lines.append(vertical_lines[-1])
        
        # 计算交叉点
        corners = []
        for h_line in outer_h_lines:
            for v_line in outer_v_lines:
                x1, y1, x2, y2 = h_line
                x3, y3, x4, y4 = v_line
                
                # 计算交点
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator != 0:
                    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
                    corners.append((int(x), int(y)))
        
        # 按顺时针顺序排序角点
        if len(corners) == 4:
            center = np.mean(corners, axis=0)
            corners.sort(key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
        
        return corners

    def create_grid_points(self, outer_corners):
        """
        根据最外圈角点创建网格点
        Args:
            outer_corners: 最外圈角点列表
        Returns:
            list: 网格点列表
        """
        if len(outer_corners) != 4:
            return []
        
        # 将角点转换为numpy数组
        corners = np.array(outer_corners, dtype=np.float32)
        
        # 创建目标网格点
        grid_points = []
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                # 计算网格点的相对位置
                x = j / (self.BOARD_SIZE - 1)
                y = i / (self.BOARD_SIZE - 1)
                
                # 使用双线性插值计算实际坐标
                p1 = corners[0] * (1-x) * (1-y)
                p2 = corners[1] * x * (1-y)
                p3 = corners[2] * x * y
                p4 = corners[3] * (1-x) * y
                
                point = p1 + p2 + p3 + p4
                grid_points.append((int(point[0]), int(point[1])))
        
        return grid_points

    def detect_board_contour(self, img):
        """
        检测棋盘轮廓，使用多种方法级联检测
        Returns:
            numpy.ndarray: 棋盘轮廓点，如果检测失败则返回None
        """
        height, width = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5,5), np.uint8)
        
        # 1. 首先尝试检测黑色边框
        mask_black = cv2.inRange(hsv, self.LOWER_BLACK, self.UPPER_BLACK)
        mask_black_opened = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_black_closed = cv2.morphologyEx(mask_black_opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        black_contours, _ = cv2.findContours(mask_black_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if black_contours:
            valid_contours = []
            min_area = width * height * 0.05
            max_area = width * height * 0.90
            for cnt in black_contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if 4 <= len(approx) <= 6:
                        valid_contours.append(cnt)
            
            if valid_contours:
                return max(valid_contours, key=cv2.contourArea)
        
        # 2. 如果黑色边框检测失败，尝试黄色边框检测
        mask_yellow = cv2.inRange(hsv, self.LOWER_YELLOW, self.UPPER_YELLOW)
        mask_yellow_closed = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_yellow_opened = cv2.morphologyEx(mask_yellow_closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask_yellow_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            board_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(board_contour)
            if area >= (width * height * 0.01):
                return board_contour
        
        # 3. 如果黑色和黄色检测都失败，尝试白色背景检测
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
        _, thresh_board = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        
        board_contours, _ = cv2.findContours(thresh_board, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        if board_contours:
            possible_boards = []
            min_area = width * height * 0.05
            max_area = width * height * 0.90
            for cnt in board_contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                    if len(approx) == 4:
                        possible_boards.append(cnt)
            
            if possible_boards:
                return max(possible_boards, key=cv2.contourArea)
        
        return None

    def detect_pieces(self, warped_board, grid_points=None):
        """
        检测棋盘上的棋子
        Args:
            warped_board (numpy.ndarray): 透视变换后的棋盘图像
            grid_points (list): 网格交点坐标列表
        Returns:
            list: 检测到的棋子列表，每个元素为((row, col), color_str)
        """
        warped_gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
        warped_gray_blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)
        
        circles = cv2.HoughCircles(warped_gray_blurred, cv2.HOUGH_GRADIENT,
                                 dp=self.HOUGH_DP, minDist=self.HOUGH_MIN_DIST,
                                 param1=self.HOUGH_PARAM1, param2=self.HOUGH_PARAM2,
                                 minRadius=self.HOUGH_MIN_RADIUS, maxRadius=self.HOUGH_MAX_RADIUS)
        
        piece_coordinates = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (px, py, r) in circles:
                if px < 0 or px >= self.WARP_SIZE or py < 0 or py >= self.WARP_SIZE:
                    continue
                
                radius_for_sampling = max(1, int(r * 0.5))
                sample_region = warped_gray[py-radius_for_sampling:py+radius_for_sampling,
                                          px-radius_for_sampling:px+radius_for_sampling]
                avg_gray = np.mean(sample_region)
                
                # 确定棋子颜色
                if avg_gray > self.PIECE_WHITE_THRESH:
                    color = "white"
                elif avg_gray < self.PIECE_BLACK_THRESH:
                    color = "black"
                else:
                    continue
                
                if grid_points is not None:
                    # 使用网格交点确定坐标
                    min_dist = float('inf')
                    closest_point = None
                    for i, (gx, gy) in enumerate(grid_points):
                        dist = np.sqrt((px - gx)**2 + (py - gy)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_point = i
                    
                    if closest_point is not None:
                        row = closest_point // self.BOARD_SIZE
                        col = closest_point % self.BOARD_SIZE
                        piece_coordinates.append(((row, col), color))
                else:
                    # 使用传统方法计算坐标
                    delta = self.WARP_SIZE / (self.BOARD_SIZE - 1)
                    col = round(px / delta)
                    row = round(py / delta)
                    
                    if 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE:
                        piece_coordinates.append(((row, col), color))
        
        return piece_coordinates

    def process_image(self, image_path):
        """
        处理输入图像，检测棋盘和棋子
        Args:
            image_path (str): 输入图像路径
        Returns:
            tuple: (棋子列表, 透视变换后的棋盘图像, 网格交点列表)
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"错误：无法加载图像 {image_path}")
            return None, None, None
        
        # 检测棋盘轮廓
        board_contour = self.detect_board_contour(img)
        if board_contour is None:
            print("错误：无法检测到棋盘轮廓")
            return None, None, None
        
        # 获取棋盘角点
        epsilon = 0.03 * cv2.arcLength(board_contour, True)
        approx_corners = cv2.approxPolyDP(board_contour, epsilon, True)
        
        if approx_corners is None or len(approx_corners) != 4:
            print(f"错误：未能精确找到棋盘轮廓的4个角点")
            return None, None, None
        
        # 透视变换
        src_pts = approx_corners.reshape(4, 2).astype("float32")
        ordered_src_pts = self.order_points(src_pts)
        
        dst_pts = np.array([
            [0, 0],
            [self.WARP_SIZE - 1, 0],
            [self.WARP_SIZE - 1, self.WARP_SIZE - 1],
            [0, self.WARP_SIZE - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(ordered_src_pts, dst_pts)
        warped_board = cv2.warpPerspective(img, M, (self.WARP_SIZE, self.WARP_SIZE))
        
        # 找到最外圈的交叉点
        outer_corners = self.find_outer_corners(warped_board)
        
        if len(outer_corners) == 4:
            # 创建网格点
            grid_points = self.create_grid_points(outer_corners)
        else:
            print("警告：未能找到所有外圈角点，使用传统方法")
            grid_points = None
        
        # 检测棋子
        pieces = self.detect_pieces(warped_board, grid_points)
        
        return pieces, warped_board, grid_points 