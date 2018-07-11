import csv
import os    
# 如果不存在目录 就创造目录
if not os.path.exists('./out/random/'):
    os.makedirs('./out/random/')
if not os.path.isfile('./out/random/print_result.csv'):
    csvfile = open('./out/random/print_result.csv', 'w')
    print('./out/random/print_result.csv'+" make flie success!!")
if not os.path.exists('./temp/lfw/result/'):
    os.makedirs('./temp/lfw/result/')
    print('./temp/lfw/result/'+" make flie success!!")
if not os.path.isfile('./temp/lfw/result/neg.csv'):
    csvfile = open('./temp/lfw/result/neg.csv', 'w')
    print('./temp/lfw/result/neg.csv'+" make flie success!!")
if not os.path.isfile('./temp/lfw/result/pos.csv'):
    csvfile = open('./temp/lfw/result/pos.csv', 'w')
    print('./temp/lfw/result/pos.csv'+" make flie success!!")
