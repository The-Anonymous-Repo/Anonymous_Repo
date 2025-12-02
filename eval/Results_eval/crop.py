import os
import cv2
import numpy as np

def crop_images_from_folders():
    """
    从7个文件夹的5个场景(Scene36-Scene40)中截取指定位置的256x256区域
    截取Dem_*.png和对应的GT_*.png
    保存一张带红色框标记的GT原始图片
    """
    
    # 获取脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 定义7个文件夹
    folders = [
        "IGRI-2",
        "IGRI-2->BM3D", 
        "Ours",
        "PFCD->IGRI-2",
        "TCPD",
        "TCPD->Unet",
        "Unet->TCPD"
    ]
    
    # 定义需要处理的场景
    scenes = ["Scene36", "Scene37", "Scene38", "Scene39", "Scene40"]
    
    # 定义每个场景的截取位置 (x, y)
    # 请在此处修改每个场景的左上角坐标
    scene_crop_coords = {
        "Scene36": (400, 400), # 示例坐标，请修改
        "Scene37": (100, 300), # 示例坐标，请修改
        "Scene38": (600, 300), # 示例坐标，请修改
        "Scene39": (766, 5),   # 之前随机生成的坐标
        "Scene40": (700, 200), # 示例坐标，请修改
    }
    
    # 需要截取的文件（截取output，原图显示对应的GT）
    target_files = [
        "Dem_R_AoP_DoP.png", 
        "Dem_G_AoP_DoP.png", 
        "Dem_B_AoP_DoP.png"
    ]
    gt_files = [
        "GT_R_AoP_DoP.png", 
        "GT_G_AoP_DoP.png", 
        "GT_B_AoP_DoP.png"
    ]
    
    # 截取尺寸
    crop_size = 256
    
    # 创建输出目录
    output_base = os.path.join(base_dir, "cropped_results")
    os.makedirs(output_base, exist_ok=True)
    
    for scene in scenes:
        print(f"\n{'='*20}")
        print(f"正在处理场景：{scene}")
        print(f"{'='*20}")
        
        # 获取该场景的截取坐标
        if scene not in scene_crop_coords:
            print(f"警告：未定义 {scene} 的截取坐标，跳过")
            continue
            
        crop_x, crop_y = scene_crop_coords[scene]
        print(f"截取位置：({crop_x}, {crop_y})")
        print(f"截取区域：({crop_x}, {crop_y}) 到 ({crop_x + crop_size}, {crop_y + crop_size})")
        
        # 对每个文件夹进行处理
        for folder in folders:
            print(f"\n  处理文件夹：{folder}")
            
            # 创建文件夹对应的输出目录 (结构: cropped_results/SceneXX/FolderXX)
            # 这样方便查看同一个场景下不同方法的对比
            folder_output_dir = os.path.join(output_base, scene, folder.replace("->", "_to_"))
            os.makedirs(folder_output_dir, exist_ok=True)
            
            # 先处理GT原图（只需要处理一次）
            gt_original_processed = False
            gt_img_with_box = None
            
            # 处理每个目标文件
            for i, target_file in enumerate(target_files):
                gt_file = gt_files[i]  # 对应的GT文件
                
                output_input_path = os.path.join(base_dir, folder, scene, target_file)
                gt_input_path = os.path.join(base_dir, folder, scene, gt_file)
                
                # 定义输出路径
                cropped_output_path = os.path.join(folder_output_dir, target_file)
                cropped_gt_path = os.path.join(folder_output_dir, f"GT_{target_file.replace('Dem_', '')}")
                
                # 检查output文件是否存在
                if not os.path.exists(output_input_path):
                    print(f"    警告：找不到output文件 {output_input_path}")
                    continue
                    
                # 检查GT文件是否存在
                if not os.path.exists(gt_input_path):
                    print(f"    警告：找不到GT文件 {gt_input_path}")
                    continue
                
                # 1. 读取并截取output图像
                output_img = cv2.imread(output_input_path)
                if output_img is None:
                    print(f"    错误：无法读取output图像 {output_input_path}")
                    continue
                
                # 检查图像尺寸
                height, width = output_img.shape[:2]
                if width < crop_x + crop_size or height < crop_y + crop_size:
                    print(f"    错误：截取区域超出图像边界 ({width}x{height})")
                    continue
                
                # 截取指定区域并保存
                cropped_output = output_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                success = cv2.imwrite(cropped_output_path, cropped_output)
                if success:
                    print(f"    成功截取并保存：{target_file}")
                else:
                    print(f"    错误：无法保存截取图像 {cropped_output_path}")
                    continue
                
                # 2. 读取GT图像并截取对应的小图
                gt_img = cv2.imread(gt_input_path)
                if gt_img is None:
                    print(f"    错误：无法读取GT图像 {gt_input_path}")
                    continue
                
                # 截取GT对应的小图
                cropped_gt = gt_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                success = cv2.imwrite(cropped_gt_path, cropped_gt)
                if success:
                    print(f"    成功截取并保存GT小图：GT_{target_file.replace('Dem_', '')}")
                else:
                    print(f"    错误：无法保存GT截取图像 {cropped_gt_path}")
                
                # 3. 处理带框的GT原图（只处理一次）
                if not gt_original_processed and i == 1: # 选择G通道 (index 1) 作为参考
                    gt_img_with_box = gt_img.copy()
                    # 绘制红色矩形框 (BGR格式，红色是 (0, 0, 255))
                    cv2.rectangle(gt_img_with_box, 
                                 (crop_x, crop_y), 
                                 (crop_x + crop_size, crop_y + crop_size), 
                                 (0, 0, 255), 
                                 thickness=3)
                    gt_original_processed = True
            
            # 保存带框的GT原始图像（只保存一次）
            if gt_img_with_box is not None:
                original_with_box_path = os.path.join(folder_output_dir, "original_with_box_GT.png")
                success = cv2.imwrite(original_with_box_path, gt_img_with_box)
                if success:
                    print(f"    成功保存带框GT原图")
                else:
                    print(f"    错误：无法保存带框GT原图 {original_with_box_path}")
            
            # 4. 处理 results 文件夹下的 GT_0.png 和 Dem_0.png
            # 路径: ../results/{folder}/{scene}/GT_0.png
            results_dir = os.path.join(base_dir, "..", "results", folder, scene)
            
            # 处理 GT_0.png (既要画框也要截取小图)
            gt_0_path = os.path.join(results_dir, "GT_0.png")
            if os.path.exists(gt_0_path):
                gt_0_img = cv2.imread(gt_0_path)
                if gt_0_img is not None:
                    # A. 截取小图
                    cropped_gt_0 = gt_0_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                    cropped_gt_0_path = os.path.join(folder_output_dir, "GT_0_crop.png")
                    success = cv2.imwrite(cropped_gt_0_path, cropped_gt_0)
                    if success:
                        print(f"    成功截取并保存GT_0小图：GT_0_crop.png")
                    
                    # B. 绘制红色矩形框并保存原图 (只保存一次)
                    # 注意：这里我们使用原图的副本画框，以免影响后续处理（虽然这里是最后一步了）
                    gt_0_with_box = gt_0_img.copy()
                    cv2.rectangle(gt_0_with_box, 
                                 (crop_x, crop_y), 
                                 (crop_x + crop_size, crop_y + crop_size), 
                                 (0, 0, 255), 
                                 thickness=3)
                    
                    gt_0_box_output_path = os.path.join(folder_output_dir, "original_with_box_GT_0.png")
                    success = cv2.imwrite(gt_0_box_output_path, gt_0_with_box)
                    if success:
                        print(f"    成功保存带框GT_0原图：original_with_box_GT_0.png")
                else:
                    print(f"    错误：无法读取GT_0图像 {gt_0_path}")
            else:
                print(f"    警告：找不到GT_0文件 {gt_0_path}")

            # 处理 Dem_0.png (截取小图)
            dem_0_path = os.path.join(results_dir, "Dem_0.png")
            if os.path.exists(dem_0_path):
                dem_0_img = cv2.imread(dem_0_path)
                if dem_0_img is not None:
                    # 截取小图
                    cropped_dem_0 = dem_0_img[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
                    cropped_dem_0_path = os.path.join(folder_output_dir, "Dem_0_crop.png")
                    success = cv2.imwrite(cropped_dem_0_path, cropped_dem_0)
                    if success:
                        print(f"    成功截取并保存Dem_0小图：Dem_0_crop.png")
                else:
                    print(f"    错误：无法读取Dem_0图像 {dem_0_path}")
            else:
                print(f"    警告：找不到Dem_0文件 {dem_0_path}")
    
    print(f"\n所有场景截取完成！结果保存在 {output_base} 目录中")

if __name__ == "__main__":
    crop_images_from_folders()
