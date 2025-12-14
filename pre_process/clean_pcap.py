import os
import pdb
import sys




def unzip (src_path, dst_path):
    entries = os.listdir(src_path)
    files = [entry for entry in entries if os.path.isfile(os.path.join(src_path, entry))]
    for file_name in files:
        os.system("unzip " + src_path + file_name + ' -d' + dst_path)





def mkdir (src_path, dst_path):
    entries = os.listdir(src_path)
    folders = [entry for entry in entries]
    for folder_name in folders:
        os.system("mkdir " + dst_path + folder_name)






def scan_folder_for_files (path):
    entries = os.listdir(path)
    files = [entry for entry in entries if os.path.isfile(os.path.join(path, entry))]
    return files


def scan_folder_for_folder (path):
    entries = os.listdir(path)
    folders = [entry for entry in entries if not os.path.isfile(os.path.join(path, entry))]
    return folders



def clean_pcap (src_name, dst_name):
    clean_option = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp and frame.len > 80 and not tls"'
    os.system('tshark -r ' + src_name + ' -Y ' + clean_option + ' -w ' + dst_name + ' > /dev/null')




def main (args):
    src_path = args[1]
    dst_path = args[2]

    # pdb.set_trace()    

    file_lst = scan_folder_for_files(src_path)
    
    k = 0
    for file in file_lst:
        print(k, len(file_lst))
        src_file = src_path + file
        dst_file = dst_path + file
        clean_pcap (src_file, dst_file)
        k += 1



if __name__ == "__main__":
    """ usage: python clean_pcap.py src/folder/path/ dst/folder/path/ 
    
    for example:
    
    # label 1-power
    python clean_pcap.py /home/KaleidoScope/CICIOT-2022/datasets/ip_tuple/1-power/ /home/KaleidoScope/CICIOT-2022/datasets/clean_pcap/1-power/

    # label 2-idle
    python clean_pcap.py /home/KaleidoScope/CICIOT-2022/datasets/ip_tuple/2-idle/2-idle-1/ /home/KaleidoScope/CICIOT-2022/datasets/clean_pcap/2-idle/2-idle-1/



    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/1/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/1/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/2/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/2/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/3/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/3/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/4/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/4/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/5/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/5/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/6/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/6/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/7/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/7/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/8/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/8/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/cross-platform/dataset/pcap/ios/9/ /mnt/winter/UTFormer_COMNET/cross-platform/dataset/raw/ios/9/

    
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/IoT2022/power_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/clean_power_pcap/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/IoT2022/interaction_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/clean_interaction_pcap/
    python clean_pcap.py /mnt/winter/UTFormer_COMNET/IoT2022/idle_pcap/ /mnt/winter/UTFormer_COMNET/IoT2022/clean_idle_pcap/

    """
    main(sys.argv)