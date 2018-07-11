import os
import shutil



if __name__ == '__main__':
    n=0
    for line in open("./MS-Celeb-1M_clean_list.txt"): 
        input_dir = './data/'  # 输入的人脸图片数据集
        output_dir = './out/'         # 输出的图片总目录 
        n=n+1
        linelist = line.split(" ")
        input_dir = input_dir+linelist[0]
        output_dir = output_dir+linelist[-1].replace("\n", "")
        name = input_dir.split("/")[-1]
        # 如果不存在目录 退出执行下一个
        if not os.path.exists(input_dir):
            print(input_dir + "   not existed!")
        else:
            # 如果不存在目录 就创造目录
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            shutil.copyfile(input_dir,output_dir+'/'+name)
            print(n)
            print(input_dir + " copyed")
           
    print('Done')
