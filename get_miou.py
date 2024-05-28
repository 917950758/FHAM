import os

from PIL import Image
from tqdm import tqdm

from FHAMN import FHAMN_
from utils.utils_metrics import compute_mIoU, show_results


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 8

    # --------------------------------------------#
    #   区分的种类
    # --------------------------------------------#
    name_classes = ["null","background", "Building", "Road", "Water", "Barren",  "Forest", "Agriculture"]
    # -------------------------------------------------------#
    #   指向数据集所在的文件夹
    # -------------------------------------------------------#
    dataset_path = r'Urban'

    # 从test.txt文件中读取图像id列表
    image_ids = open(os.path.join(dataset_path, "./test.txt"), 'r').read().splitlines()

    # 设置ground truth和预测结果的路径
    gt_dir = os.path.join(dataset_path, "./masks_png/")
    miou_out_path = r"Remote data/new_da_aspp/MAX+mult+glo"
    pred_dir = os.path.join(miou_out_path, './detection-results')

    # 如果miou_mode为0或1，则进行预测结果的保存
    if miou_mode == 0 or miou_mode == 1:
        # 如果预测结果保存路径不存在，则创建该路径
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        # 加载模型
        print("Load model.")
        net = FHAMN_()
        print("Load model done.")

        print("Get predict result.")
        # 遍历每个图像id，获取预测结果
        for image_id in tqdm(image_ids):
            # 读取图像
            image_path = os.path.join(dataset_path, "./images_png/" + image_id + ".png")
            image = Image.open(image_path)
            # 使用网络模型获取预测结果
            image = net.get_miou_png(image)
            # 保存预测结果
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    # 如果miou_mode为0或2，则计算mIoU指标
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # 执行计算mIoU的函数，获取结果
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)
        print("Get miou done.")
        # 显示mIoU结果
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
