# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
from chess_detector import ChessDetector

def visualize_board(warped_board, pieces, grid_points=None, outer_corners=None, board_size=15):
    """
    可视化棋盘和棋子
    Args:
        warped_board: 透视变换后的棋盘图像
        pieces: 检测到的棋子列表
        grid_points: 网格点列表
        outer_corners: 最外圈角点列表
        board_size: 棋盘尺寸
    """
    # 创建可视化图像
    vis_img = warped_board.copy()
    
    if grid_points is not None:
        # 绘制网格点
        for point in grid_points:
            cv2.circle(vis_img, point, 2, (0, 255, 0), -1)
        
        # 绘制网格线
        for i in range(board_size):
            # 绘制水平线
            for j in range(board_size-1):
                p1 = grid_points[i*board_size + j]
                p2 = grid_points[i*board_size + j + 1]
                cv2.line(vis_img, p1, p2, (0, 0, 255), 1)
            
            # 绘制垂直线
            for j in range(board_size-1):
                p1 = grid_points[j*board_size + i]
                p2 = grid_points[(j+1)*board_size + i]
                cv2.line(vis_img, p1, p2, (0, 0, 255), 1)
    
    if outer_corners is not None and len(outer_corners) == 4:
        # 绘制最外圈角点
        for point in outer_corners:
            cv2.circle(vis_img, point, 5, (255, 0, 0), -1)
        
        # 绘制最外圈线
        for i in range(4):
            cv2.line(vis_img, outer_corners[i], outer_corners[(i+1)%4], (255, 0, 0), 2)
    
    # 绘制棋子
    for (row, col), color in pieces:
        if grid_points is not None:
            # 使用网格点确定棋子位置
            point_index = row * board_size + col
            if point_index < len(grid_points):
                center_x, center_y = grid_points[point_index]
            else:
                continue
        else:
            # 使用传统方法计算棋子位置
            delta = warped_board.shape[0] / (board_size - 1)
            center_x = int(col * delta)
            center_y = int(row * delta)
        
        radius = int(warped_board.shape[0] / (board_size * 2.5))  # 棋子半径
        
        if color == "black":
            cv2.circle(vis_img, (center_x, center_y), radius, (0, 0, 0), -1)
            cv2.circle(vis_img, (center_x, center_y), radius, (255, 255, 255), 1)
        else:  # white
            cv2.circle(vis_img, (center_x, center_y), radius, (255, 255, 255), -1)
            cv2.circle(vis_img, (center_x, center_y), radius, (0, 0, 0), 1)
    
    return vis_img

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='五子棋棋盘检测程序')
    parser.add_argument('image_path', type=str, help='要处理的图片路径')
    parser.add_argument('--board-size', type=int, default=15, help='棋盘尺寸，默认为15')
    parser.add_argument('--warp-size', type=int, default=750, help='透视变换后图像的边长，默认为750')
    parser.add_argument('--output-prefix', type=str, default='', help='输出文件的前缀，默认为空')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建检测器实例
    detector = ChessDetector(board_size=args.board_size, warp_size=args.warp_size)
    
    # 处理图像
    pieces, warped_board, grid_points = detector.process_image(args.image_path)
    
    if pieces is not None and warped_board is not None:
        print(f"成功检测到 {len(pieces)} 个棋子")
        for (row, col), color in pieces:
            print(f"在位置 ({row}, {col}) 发现{color}棋子")
        
        # 获取最外圈角点
        outer_corners = detector.find_outer_corners(warped_board)
        
        # 可视化结果
        vis_img = visualize_board(warped_board, pieces, grid_points, outer_corners)
        
        # 显示结果
        cv2.imshow('Original Board', cv2.imread(args.image_path))
        cv2.imshow('Warped Board', warped_board)
        cv2.imshow('Visualization', vis_img)
        
        # 保存结果
        output_prefix = args.output_prefix if args.output_prefix else 'output'
        cv2.imwrite(f'{output_prefix}_warped_board.jpg', warped_board)
        cv2.imwrite(f'{output_prefix}_visualization.jpg', vis_img)
        
        print(f"结果已保存为 {output_prefix}_warped_board.jpg 和 {output_prefix}_visualization.jpg")
        
        # 等待按键
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("处理失败，请检查图像和参数设置")

if __name__ == "__main__":
    main()
