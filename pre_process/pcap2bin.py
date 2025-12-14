from scapy.all import *
import os
from clean_pcap import scan_folder_for_files, scan_folder_for_folder
import pdb


""" 
    输入：pcap文件
    输出：bin文件，并对每个pkt的相关字段进行截断，并拼接前n个字节的payload，最后对齐等长
    粒度：一个流一个txt文件，一个包最终保留1500Byte
"""





def flow2bin_all_ip (src_file, dst_file):
    # 保留完整的ip包

    kept_length = 1500
    packets = rdpcap(src_file)

    # 遍历文件中的每个数据包
    pkts_context = ''
    for packet in packets:
        if packet.haslayer('IP'):  
        # totally 3 bytes from ip header
            ip_header = packet['IP']
            
            # 获取IP头部的原始二进制数据
            ip_header_raw = ip_header.build()
            ip_byte = ip_header_raw[0:]
            # ip_byte_hex = ip_byte.hex()

        
            if packet.haslayer('TCP') or packet.haslayer('UDP'):
                
                if(len(ip_byte) >= kept_length):
                    ip_byte_hex = ip_byte[0: kept_length].hex()
                else:
                    ip_byte_hex = ip_byte.ljust(kept_length, b'\x00').hex()

                pkt_context = ip_byte_hex

            pkts_context += pkt_context + '\n'

    dst_file_handler = open(dst_file, 'w')
    dst_file_handler.write(pkts_context)
    dst_file_handler.close()
    
    


def main (args):
    src_path = args[1]
    dst_path = args[2]
    # 得到src_path下所有的sub_folder

    i = 0
    
    files = scan_folder_for_files(src_path)
    for file in files:
        i += 1
        print(i, len(files), file)

        src_file = src_path + file
        dst_file = dst_path + file[0:-5] + '.txt'
        flow2bin_all_ip (src_file, dst_file)
        


if __name__ == "__main__":
    """ usage: python clean_pcap.py src/folder/path/ dst/folder/path/ 
    
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/1/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/1/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/2/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/2/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/3/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/3/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/4/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/4/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/5/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/5/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/6/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/6/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/7/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/7/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/android/8/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/android/8/
    
    android datasets has 15*7+16=121class

    
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/1/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/1/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/2/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/2/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/3/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/3/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/4/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/4/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/5/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/5/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/6/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/6/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/7/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/7/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/8/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/8/
    
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/9/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/bin_1/ios/9/



    python pcap2bin.py /mnt/winter/UTFormer_COMNET/IoT2022/clean_idle_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/idle_bin/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/IoT2022/clean_power_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/power_bin/
    python pcap2bin.py /mnt/winter/UTFormer_COMNET/IoT2022/clean_interaction_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/interaction_bin/

    
    
    """
    main(sys.argv)