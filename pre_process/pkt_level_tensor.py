import os
import pdb
from clean_pcap import scan_folder_for_files, scan_folder_for_folder
from scapy.all import *
import torch


""" 
    源数据folder: pkt_level_dataset
    目标数据folder: pkt_level_tensor
    将包粒度数据从hex string 转为tensor
"""




def merge2txt (args):
    # 将所有的不同class的原始数据合并到1个数据txt和1个label txt里
    src_path = args[1]
    dst_path = args[2]
    # 得到src_path下所有的sub_folder
    file_lst = scan_folder_for_files(src_path)

    train_dataset = open(dst_path+'train_malware.txt', 'w')
    train_label = open(dst_path+'train_malware_label.txt', 'w')
    label = '1'


    n_pkt = 0
    for file in file_lst:
        file_handler = open(src_path + '/' + file, 'r')
        for line in file_handler.readlines():
            train_dataset.write(line)
            train_label.writelines(label+'\n')
            n_pkt += 1
        file_handler.close()

        print('n_pkt: ', n_pkt)

    train_dataset.close()
    train_label.close()






def merge2txt_non_auto ():
    # 将所有的不同class的原始数据合并到1个数据txt和1个label txt里
    src_file = [
        '/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train_normal_label.txt',
        '/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train_malware_label.txt'
    ]
    dst_path = '/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/'

    txt_dataset = open(dst_path+'train_dataset_label.txt', 'w')
    label = '0'


    n_pkt = 0
    for file in src_file:
        file_handler = open(file, 'r')
        for line in file_handler.readlines():
            txt_dataset.write(line)
            n_pkt += 1
        file_handler.close()

        print('n_pkt: ', n_pkt)

    txt_dataset.close()




def cat_dataset (args):
    # 从三个txt中选取不同的文本进行拼接
    # split dataset into train and teste

    src_file_1 = args[1]
    src_file_2 = args[2]
    src_file_3 = args[3]
    dst_file = args[4]
    option = args[5]

    train_size = [130870, 196305, 65435]
    test_size = [43624, 65436, 21812]
    n_sample = 0
    dst_file_handler = open(dst_file, 'w')

    if(option == 'train'):
        src_file_handler = open(src_file_1, 'r')
        for line in src_file_handler.readlines():
            dst_file_handler.write(line)
            n_sample += 1
            if(n_sample == train_size[0]):
                break
        src_file_handler.close()
        n_sample = 0

        src_file_handler = open(src_file_2, 'r')
        for line in src_file_handler.readlines():
            dst_file_handler.write(line)
            n_sample += 1
            if(n_sample == train_size[1]):
                break
        src_file_handler.close()
        n_sample = 0

        src_file_handler = open(src_file_3, 'r')
        for line in src_file_handler.readlines():
            dst_file_handler.write(line)
            n_sample += 1
            if(n_sample == train_size[2]):
                break
        src_file_handler.close()
        n_sample = 0

        dst_file_handler.close()
    

    if(option == 'test'):
        src_file_handler = open(src_file_1, 'r')
        for line in src_file_handler.readlines():
            if(n_sample < train_size[0]):
                n_sample += 1
                continue
            if(n_sample >= train_size[0] and n_sample < train_size[0] + test_size[0]):
                dst_file_handler.write(line)
                n_sample += 1
            if(n_sample > train_size[0] + test_size[0]):
                break
        src_file_handler.close()
        n_sample = 0


        src_file_handler = open(src_file_2, 'r')
        for line in src_file_handler.readlines():
            if(n_sample < train_size[1]):
                n_sample += 1
                continue
            if(n_sample >= train_size[1] and n_sample < train_size[1] + test_size[1]):
                dst_file_handler.write(line)
                n_sample += 1
            if(n_sample > train_size[1] + test_size[1]):
                break
        src_file_handler.close()
        n_sample = 0


        src_file_handler = open(src_file_3, 'r')
        for line in src_file_handler.readlines():
            if(n_sample < train_size[2]):
                n_sample += 1
                continue
            if(n_sample >= train_size[2] and n_sample < train_size[2] + test_size[2]):
                dst_file_handler.write(line)
                n_sample += 2
            if(n_sample > train_size[1] + test_size[2]):
                break
        src_file_handler.close()
        n_sample = 0

        dst_file_handler.close()



    
        
    


def txt2tensor ():
    """ 
        将hex string数据转化为tensor格式
    """

    train_txt_dataset = open('/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train_dataset.txt', 'r')
    train_txt_label = open('/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train_dataset_label.txt', 'r')
    test_txt_dataset = open('/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/test_dataset.txt', 'r')
    test_txt_label = open('/home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/test_dataset_label.txt', 'r')


    # 训练数据集
    train_list_dataset = []
    for hex_str in train_txt_dataset.readlines():
        # 去除可能存在的换行符并padding
        hex_str = hex_str[0:-1]
        if(len(hex_str) % 2 == 1):
            hex_str += '0'

        # 分割
        hex_chunks = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        # 将每个十六进制数转换为无符号整数
        unsigned_integers = [int(chunk, 16) for chunk in hex_chunks]
        train_list_dataset.append(unsigned_integers)

    train_tensor_dataset = torch.tensor(train_list_dataset)
    train_tensor_dataset = train_tensor_dataset.float() / 256
    torch.save(train_tensor_dataset, '/home/KaleidoScope/CICIDS-2017/dataset/tensor_data/train_tensor_dataset.pt')


    # 测试数据集
    test_list_dataset = []
    for hex_str in test_txt_dataset.readlines():
        # 去除可能存在的换行符并padding
        hex_str = hex_str[0:-1]
        if(len(hex_str) % 2 == 1):
            hex_str += '0'

        # 分割
        hex_chunks = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
        # 将每个十六进制数转换为无符号整数
        unsigned_integers = [int(chunk, 16) for chunk in hex_chunks]
        test_list_dataset.append(unsigned_integers)

    test_tensor_dataset = torch.tensor(test_list_dataset)
    test_tensor_dataset = test_tensor_dataset.float() / 256
    torch.save(test_tensor_dataset, '/home/KaleidoScope/CICIDS-2017/dataset/tensor_data/test_tensor_dataset.pt')



    # 训练label
    train_list_label = []
    for label in train_txt_label.readlines():
        train_list_label.append(int(label))

    train_tensor_label = torch.tensor(train_list_label)
    train_tensor_label = train_tensor_label.long()
    torch.save(train_tensor_label, '/home/KaleidoScope/CICIDS-2017/dataset/tensor_data/train_tensor_label.pt')

    # 测试label
    test_list_label = []
    for label in test_txt_label.readlines():
        test_list_label.append(int(label))

    test_tensor_label = torch.tensor(test_list_label)
    test_tensor_label = test_tensor_label.long()
    torch.save(test_tensor_label, '/home/KaleidoScope/CICIDS-2017/dataset/tensor_data/test_tensor_label.pt')









def txt2tensor_wsplit (src_path, dst_path, split_r=0.7):
    """ 
        将hex string数据转化为tensor格式并自动划分好训练集和测试集
    """


    src_data_folder = src_path + '/data'
    src_label_folder = src_path + '/label'
    dst_train_data_folder = dst_path + '/train_data'
    dst_test_data_folder = dst_path + '/test_data'
    dst_train_label_folder = dst_path + '/train_label'
    dst_test_label_folder = dst_path + '/test_label'


    # 处理数据
    idf = 0
    file_lst = scan_folder_for_files(src_data_folder)
    for file in file_lst:
        src_file_handler = open(src_data_folder + '/' + file)
        txt_content = src_file_handler.readlines()
        line_count = len(txt_content)
        n_train_thrh = int(line_count * split_r)

        train_dataset = []
        test_dataset = []
        n_train = 0

        print(idf, len(file_lst), file)

        # 逐行/报文转tensor并划分数据集
        for hex_str in txt_content:
            # 去除可能存在的换行符并padding
            hex_str = hex_str[0:-1]
            if(len(hex_str) % 2 == 1):
                hex_str += '0'
            # 补齐到1500 Byte
            while(len(hex_str) < 3000):
                hex_str += '0'
            # 分割
            hex_chunks = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
            # 将每个十六进制数转换为无符号整数
            unsigned_integers = [int(chunk, 16) for chunk in hex_chunks]

            # 根据比例划分数据
            if(n_train < n_train_thrh):
                train_dataset.append(unsigned_integers)
            else:
                test_dataset.append(unsigned_integers)
            n_train += 1
            
        
        train_tensor_dataset = torch.tensor(train_dataset)
        train_tensor_dataset = train_tensor_dataset.float() / 256
        
        test_tensor_dataset = torch.tensor(test_dataset)
        test_tensor_dataset = test_tensor_dataset.float() / 256

        idf += 1
        filename = file[0:-5]

        torch.save(train_tensor_dataset, dst_train_data_folder + '/' + filename + '.pt')
        torch.save(test_tensor_dataset, dst_test_data_folder + '/' + filename + '.pt')


    # 处理label
    file_lst = scan_folder_for_files(src_label_folder)
    for file in file_lst:
        src_file_handler = open(src_label_folder + '/' + file)
        txt_content = src_file_handler.readlines()
        line_count = len(txt_content)
        n_train_thrh = int(line_count * split_r)

        train_label = []
        test_label = []
        n_train = 0
        for label in txt_content:
            # 根据比例划分标签
            if(n_train < n_train_thrh):
                train_label.append(int(label))
            else:
                test_label.append(int(label))
            n_train += 1
        
        train_tensor_dataset = torch.tensor(train_label)
        test_tensor_dataset = torch.tensor(test_label)

        filename = file[0:-5]

        torch.save(train_tensor_dataset, dst_train_label_folder + '/' + filename + '.pt')
        torch.save(test_tensor_dataset, dst_test_label_folder + '/' + filename + '.pt')















def txt2tensor_split_merge_label (src_path, dst_path, start_label, filename, split_r=0.7):
    """ 
        将hex string数据转化为tensor格式并自动划分好训练集和测试集，同时自动打标签
        start label: 确定输入src_path中的第一个label从哪开始
    """


    src_data_folder = src_path
    dst_train_data_folder = dst_path + '/'
    dst_test_data_folder = dst_path + '/'
    dst_train_label_folder = dst_path + '/'
    dst_test_label_folder = dst_path + '/'


    train_dataset = []
    test_dataset = []
    train_label = []
    test_label = []
    label = int(start_label)


    # 处理数据
    idf = 0
    file_lst = scan_folder_for_files(src_data_folder)

    for file in file_lst:
        src_file_handler = open(src_data_folder + '/' + file)
        txt_content = src_file_handler.readlines()
        line_count = len(txt_content)
        n_train_thrh = int(line_count * split_r)

        n_train = 0

        print(idf, len(file_lst), file)

        # 逐行/报文转tensor并划分数据集
        for hex_str in txt_content:
            # 去除可能存在的换行符并padding
            hex_str = hex_str[0:-1]
            if(len(hex_str) % 2 == 1):
                hex_str += '0'
            # 补齐到1500 Byte
            while(len(hex_str) < 3000):
                hex_str += '0'
            # 分割
            hex_chunks = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
            # 将每个十六进制数转换为无符号整数
            unsigned_integers = [int(chunk, 16) for chunk in hex_chunks]

            # 根据比例划分数据 并生成标签
            if(n_train < n_train_thrh):
                train_dataset.append(unsigned_integers)
                train_label.append(label)
            else:
                test_dataset.append(unsigned_integers)
                test_label.append(label)
            n_train += 1

        idf += 1
        # label += 1
        
    train_tensor_dataset = torch.tensor(train_dataset)
    train_tensor_dataset = train_tensor_dataset.float() / 256
    
    test_tensor_dataset = torch.tensor(test_dataset)
    test_tensor_dataset = test_tensor_dataset.float() / 256

    train_tensor_label = torch.tensor(train_label)
    test_tensor_label = torch.tensor(test_label)


    torch.save(train_tensor_dataset, dst_train_data_folder + '/' + filename + '_train_data.pt')
    torch.save(test_tensor_dataset, dst_test_data_folder + '/' + filename + '_test_data.pt')
    torch.save(train_tensor_label, dst_train_label_folder + '/' + filename + '_train_label.pt')
    torch.save(test_tensor_label, dst_test_label_folder + '/' + filename + '_test_label.pt')


    print("next start label: ", len(file_lst)+int(start_label))











def merge_tensor ():
    # 最普通的tensor融合：融合一个文件夹下的所有tensor到一个.pt中

    src_path = '/mnt/winter/UTFormer_COMNET/IoT2022/test_label/'
    dst_file = '/mnt/winter/UTFormer_COMNET/IoT2022/test_label.pt'
    
    file_lst = scan_folder_for_files(src_path)
    data_tensor = []

    idf = 0
    for file in file_lst:
        print(file, idf, len(file_lst))
        tmp = torch.load(src_path + '/' + file)
        data_tensor.append(tmp)
        idf += 1

    stack_tensor = torch.cat(data_tensor, dim=0)
    torch.save(stack_tensor, dst_file)







def merge_data_label ():
    # 把对应的data和label一起融合，防止文件加载不一致的问题

    src_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_data'
    src_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_label'
    dst_data_file = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_data.pt'
    dst_label_file = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor_1/ios/test_label.pt'

    # src_data_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor/ios/train_data'
    # src_label_path = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor/ios/train_label'
    # dst_data_file = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor/ios/train_data.pt'
    # dst_label_file = '/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor/ios/train_label.pt'
    
    data_file_lst = scan_folder_for_files(src_data_path)
    label_file_lst = scan_folder_for_files(src_label_path)
    data_tensor = []
    label_tensor = []

    idf = 0
    for file in data_file_lst:
        print(file, idf, len(data_file_lst))
        
        # 处理data
        data_pt = torch.load(src_data_path + '/' + file)
        data_tensor.append(data_pt)

        # 处理label
        idx = file[0:5]
        for label_file in label_file_lst:
            if idx in label_file:
                label_pt = torch.load(src_label_path + '/' + label_file)
                label_tensor.append(label_pt)

        idf += 1

    data_stack_tensor = torch.cat(data_tensor, dim=0)
    label_stack_tensor = torch.cat(label_tensor, dim=0)

    torch.save(data_stack_tensor, dst_data_file)
    torch.save(label_stack_tensor, dst_label_file)







def main(args):
    src_path = args[1]
    dst_path = args[2]
    start_label = args[3]
    filename = args[4]
    txt2tensor_split_merge_label(src_path, dst_path, start_label, filename)




if __name__ == "__main__":
    """ usage: python pkt_level_tensor.py src/folder/path/ dst/folder/path/ 
    
    # merge txt
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/hex_pcap/3-interactions/3-interactions/ /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/3-interactions/
    
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/hex_pcap/1-power/1-power/ /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/

    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/hex_pcap/2-idle/2-idle-up-1/2-idle-1/ /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/2-tmp/


    # split dataset into train and teste
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/1-power/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/2-tmp/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/3-interactions/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/train_data.txt train
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/1-power/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/2-tmp/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/3-interactions/raw_pcap.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/test_data.txt test
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/1-power/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/2-tmp/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/3-interactions/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/train_label.txt train
    python pkt_level_tensor.py /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/1-power/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/2-tmp/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/3-interactions/raw_pcap_label.txt /home/KaleidoScope/CICIOT-2022/datasets/split_set_pre/test_label.txt test
    


    python pkt_level_tensor.py /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train/normal/ /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/
    python pkt_level_tensor.py /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/test/normal/ /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/
    python pkt_level_tensor.py /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/train/malware/ /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/
    python pkt_level_tensor.py /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/test/malware/ /home/KaleidoScope/CICIDS-2017/dataset/txt_dataset/

    """
    # cat_dataset(sys.argv)
    # merge2txt(sys.argv)
    # txt2tensor()

    # merge2txt_non_auto()


    # txt2tensor_wsplit (
    #     src_path='/mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/1',
    #     dst_path='/mnt/winter/UTFormer_COMNET/cross-platform/dataset/tensor/android/1'
    # )

    

    """

    python pkt_level_tensor.py ../dataset/bin_1/android/1 ../dataset/tensor_1/android/ 0 sub_1
    python pkt_level_tensor.py ../dataset/bin_1/android/2 ../dataset/tensor_1/android/ 15 sub_2
    python pkt_level_tensor.py ../dataset/bin_1/android/3 ../dataset/tensor_1/android/ 30 sub_3
    python pkt_level_tensor.py ../dataset/bin_1/android/4 ../dataset/tensor_1/android/ 45 sub_4
    python pkt_level_tensor.py ../dataset/bin_1/android/5 ../dataset/tensor_1/android/ 60 sub_5
    python pkt_level_tensor.py ../dataset/bin_1/android/6 ../dataset/tensor_1/android/ 75 sub_6
    python pkt_level_tensor.py ../dataset/bin_1/android/7 ../dataset/tensor_1/android/ 90 sub_7
    python pkt_level_tensor.py ../dataset/bin_1/android/8 ../dataset/tensor_1/android/ 105 sub_8

    android datasets has 15*7+16=121class


    python pkt_level_tensor.py ../dataset/bin_1/ios/1 ../dataset/tensor_1/ios/ 0 sub_1
    python pkt_level_tensor.py ../dataset/bin_1/ios/2 ../dataset/tensor_1/ios/ 20 sub_2
    python pkt_level_tensor.py ../dataset/bin_1/ios/3 ../dataset/tensor_1/ios/ 36 sub_3
    python pkt_level_tensor.py ../dataset/bin_1/ios/4 ../dataset/tensor_1/ios/ 64 sub_4
    python pkt_level_tensor.py ../dataset/bin_1/ios/5 ../dataset/tensor_1/ios/ 89 sub_5
    python pkt_level_tensor.py ../dataset/bin_1/ios/6 ../dataset/tensor_1/ios/ 92 sub_6
    python pkt_level_tensor.py ../dataset/bin_1/ios/7 ../dataset/tensor_1/ios/ 106 sub_7
    python pkt_level_tensor.py ../dataset/bin_1/ios/8 ../dataset/tensor_1/ios/ 107 sub_8
    python pkt_level_tensor.py ../dataset/bin_1/ios/9 ../dataset/tensor_1/ios/ 137 sub_9


    python pkt_level_tensor.py /mnt/winter/UTFormer_COMNET/IoT2022/idle_bin /mnt/winter/UTFormer_COMNET/IoT2022/idle_tensor 0 idle
    python pkt_level_tensor.py /mnt/winter/UTFormer_COMNET/IoT2022/power_bin /mnt/winter/UTFormer_COMNET/IoT2022/power_tensor 1 power
    python pkt_level_tensor.py /mnt/winter/UTFormer_COMNET/IoT2022/interaction_bin /mnt/winter/UTFormer_COMNET/IoT2022/interaction_tensor 2 interaction


    """

    # main(sys.argv)





    # 最终的tensor融合
    # merge_data_label()

    merge_tensor()