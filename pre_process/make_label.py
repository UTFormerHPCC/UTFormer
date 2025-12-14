from clean_pcap import scan_folder_for_files, scan_folder_for_folder
import os
import pdb
import sys




def make_label (args):
    """ 
        start label: 确定该函数中的第一个label从哪开始
    """

    src_path = args[1]
    dst_path = args[2]
    label = int(args[3])
    file_lst = scan_folder_for_files(src_path)

    idx = 0
    for file in file_lst:
        src_file_handler = open(src_path + '/' + file, 'r')
        dst_file_handler = open(dst_path + '/' + file, 'w')

        line_count = len(src_file_handler.readlines())

        
        for i in range (line_count):
            dst_file_handler.write(str(label)+'\n')

        src_file_handler.close()
        dst_file_handler.close()
        label += 1

        print(idx, len(file_lst), file)
        idx += 1



if __name__ == "__main__":

    """ 
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/1/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/1/label 0
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/2/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/2/label 15
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/3/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/3/label 30
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/4/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/4/label 45
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/5/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/5/label 60
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/6/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/6/label 75
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/7/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/7/label 90
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/8/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/8/label 105
    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/9/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/android/9/label 120
    

    python make_label.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/ios/1/data  /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin/ios/1/label 0
    """

    make_label(sys.argv)