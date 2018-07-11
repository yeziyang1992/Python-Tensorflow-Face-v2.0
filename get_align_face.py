import multiprocessing
import os

from face_lib import my_api

input_dir = './out'  # 输入的人脸图片数据集
output_dir = './train_faces'  # 输出的图片总目录
# 如果不存在目录 就创造目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


if __name__ == '__main__':
    get_face = my_api.GetAlignedFace(input_dir, output_dir)
    # 开启多进程
    pool = multiprocessing.Pool(processes=6)
    photo_names = get_face.pic_names
    print(photo_names)
    pic_num = 0  # 已存在的图片目录总数

    for file_name in photo_names:
        name = file_name
        pool.apply_async(get_face.photo_read, (file_name, pic_num))
        pic_num = pic_num + 1

    pool.close()
    pool.join()  # 调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束
    print('Done')






