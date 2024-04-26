import subprocess
import os
from pathlib import Path
def split_pcap(input_file, split_count,output_file):
    # 创建存放分割文件的文件夹
    split_files_folder = 'split_data'
    os.makedirs(split_files_folder, exist_ok=True)
    # 使用 editcap 命令进行分割
    output_prefix = os.path.join(split_files_folder, output_file)
    subprocess.run(['editcap', '-c', str(split_count), input_file, output_prefix])

folder = Path("data")
for i,pcap in enumerate(folder.iterdir()):
    name = pcap.name.split(".")[0]
    input_file = pcap
    split_count = 10000
    split_pcap(input_file, split_count, name + ".pcap")
    print(i,name,"success")

















