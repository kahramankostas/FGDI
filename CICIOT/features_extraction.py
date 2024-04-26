from scapy.layers.l2 import Ether
from scapy.all import*
from scapy.layers.inet import IP, UDP
import numpy as np
from scapy.compat import raw
from scapy.utils import PcapReader

def get_file_name_1(file_path):
    file_names = os.listdir(file_path)
    return file_names

dic={
"1c:fe:2b:98:16:dd":"Amazon Alexa Echo Dot 1",
"a0:d0:dc:c4:08:ff":"Amazon Alexa Echo Dot 2",
"1c:12:b0:9b:0c:ec":"Amazon Alexa Echo Spot",
"08:7c:39:ce:6e:2a":"Amazon Alexa Echo Studio",
"cc:f4:11:9c:d0:00":"Google Nest Mini",
"48:a6:b8:f9:1b:88":"Sonos One Speaker",
"9c:8e:cd:1d:ab:9f":"AMCREST WiFi Camera",
"3c:37:86:6f:b9:51":"Arlo Base Station",
"40:5d:82:35:14:c8":"Arlo Q Camera",
"c0:e7:bf:0a:79:d1":"Borun Sichuan-AI Camera",
"b0:c5:54:59:2e:99":"DCS8000LHA1 D-Link Mini Camera",
"44:01:bb:ec:10:4a":"HeimVision Smart WiFi Camera",
"34:75:63:73:f3:36":"Home Eye Camera",
"7c:a7:b0:cd:18:32":"Luohe Cam Dog",
"44:bb:3b:00:39:07":"Nest Indoor Camera",
"70:ee:50:68:0e:32":"Netatmo Camera",
"10:2c:6b:1b:43:be":"SIMCAM 1S (AMPAKTec)",
"b8:5f:98:d0:76:e6":"Amazon Plug",
"68:57:2d:56:ac:47":"Atomi Coffee Maker",
"8c:85:80:6c:b6:47":"Eufy HomeBase 2",
"50:02:91:b1:68:0c":"Globe Lamp ESP B1680C",
"b8:f0:09:03:9a:af":"Gosund ESP 039AAF Socket",
"b8:f0:09:03:29:79":"Gosund ESP 032979 Plug",
"50:02:91:10:09:8f":"Gosund ESP 10098F Socket",
"c4:dd:57:0c:39:94":"Gosund ESP 0C3994 Plug",
"50:02:91:1a:ce:e1":"Gosund ESP 1ACEE1  Socket",
"24:a1:60:14:7f:f9":"Gosund ESP 147FF9 Plug",
"50:02:91:10:ac:d8":"Gosund ESP 10ACD8 Plug",
"d4:a6:51:30:64:b7":"HeimVision SmartLife Radio Lamp",
"00:17:88:60:d6:4f":"Philips Hue Bridge",
"b0:09:da:3e:82:6c":"Ring Base Station AC 1236",
"50:14:79:37:80:18":"iRobot Roomba",
"00:02:75:f6:e3:cb":"Smart Board",
"d4:a6:51:76:06:64":"Teckin Plug 1",
"d4:a6:51:78:97:4e":"Teckin Plug 2",
"d4:a6:51:20:91:d1":"Yutron Plug 1",
"d4:a6:51:21:6c:29":"Yutron Plug 2",
"f0:b4:d2:f9:60:95":"D-Link DCHS-161 Water Sensor",
"ac:f1:08:4e:00:82":"LG Smart TV",
"70:ee:50:6b:a8:1a":"Netatmo Weather Station"}
dev_count = {'Amazon Alexa Echo Dot 1': 0, 'Amazon Alexa Echo Dot 2': 0, 'Amazon Alexa Echo Spot': 0, 'Amazon Alexa Echo Studio': 0, 'Google Nest Mini': 0, 'Sonos One Speaker': 0, 'AMCREST WiFi Camera': 0, 'Arlo Base Station': 0, 'Arlo Q Camera': 0, 'Borun Sichuan-AI Camera': 0, 'DCS8000LHA1 D-Link Mini Camera': 0, 'HeimVision Smart WiFi Camera': 0, 'Home Eye Camera': 0, 'Luohe Cam Dog': 0, 'Nest Indoor Camera': 0, 'Netatmo Camera': 0, 'SIMCAM 1S (AMPAKTec)': 0, 'Amazon Plug': 0, 'Atomi Coffee Maker': 0, 'Eufy HomeBase 2': 0, 'Globe Lamp ESP B1680C': 0, 'Gosund ESP 039AAF Socket': 0, 'Gosund ESP 032979 Plug': 0, 'Gosund ESP 10098F Socket': 0, 'Gosund ESP 0C3994 Plug': 0, 'Gosund ESP 1ACEE1  Socket': 0, 'Gosund ESP 147FF9 Plug': 0, 'Gosund ESP 10ACD8 Plug': 0, 'HeimVision SmartLife Radio Lamp': 0, 'Philips Hue Bridge': 0, 'Ring Base Station AC 1236': 0, 'iRobot Roomba': 0, 'Smart Board': 0, 'Teckin Plug 1': 0, 'Teckin Plug 2': 0, 'Yutron Plug 1': 0, 'Yutron Plug 2': 0, 'D-Link DCHS-161 Water Sensor': 0, 'LG Smart TV': 0, 'Netatmo Weather Station': 0}
files_add = get_file_name_1("split_data")
mac_list = ['1c:fe:2b:98:16:dd', 'a0:d0:dc:c4:08:ff', '1c:12:b0:9b:0c:ec', '08:7c:39:ce:6e:2a', 'cc:f4:11:9c:d0:00', '48:a6:b8:f9:1b:88', '9c:8e:cd:1d:ab:9f', '3c:37:86:6f:b9:51', '40:5d:82:35:14:c8', 'c0:e7:bf:0a:79:d1', 'b0:c5:54:59:2e:99', '44:01:bb:ec:10:4a', '34:75:63:73:f3:36', '7c:a7:b0:cd:18:32', '44:bb:3b:00:39:07', '70:ee:50:68:0e:32', '10:2c:6b:1b:43:be', 'b8:5f:98:d0:76:e6', '68:57:2d:56:ac:47', '8c:85:80:6c:b6:47', '50:02:91:b1:68:0c', 'b8:f0:09:03:9a:af', 'b8:f0:09:03:29:79', '50:02:91:10:09:8f', 'c4:dd:57:0c:39:94', '50:02:91:1a:ce:e1', '24:a1:60:14:7f:f9', '50:02:91:10:ac:d8', 'd4:a6:51:30:64:b7', '00:17:88:60:d6:4f', 'b0:09:da:3e:82:6c', '50:14:79:37:80:18', '00:02:75:f6:e3:cb', 'd4:a6:51:76:06:64', 'd4:a6:51:78:97:4e', 'd4:a6:51:20:91:d1', 'd4:a6:51:21:6c:29', 'f0:b4:d2:f9:60:95', 'ac:f1:08:4e:00:82', '70:ee:50:6b:a8:1a']
dev_name = ['Amazon Alexa Echo Dot 1', 'Amazon Alexa Echo Dot 2', 'Amazon Alexa Echo Spot', 'Amazon Alexa Echo Studio', 'Google Nest Mini', 'Sonos One Speaker', 'AMCREST WiFi Camera', 'Arlo Base Station', 'Arlo Q Camera', 'Borun Sichuan-AI Camera', 'DCS8000LHA1 D-Link Mini Camera', 'HeimVision Smart WiFi Camera', 'Home Eye Camera', 'Luohe Cam Dog', 'Nest Indoor Camera', 'Netatmo Camera', 'SIMCAM 1S (AMPAKTec)', 'Amazon Plug', 'Atomi Coffee Maker', 'Eufy HomeBase 2', 'Globe Lamp ESP B1680C', 'Gosund ESP 039AAF Socket', 'Gosund ESP 032979 Plug', 'Gosund ESP 10098F Socket', 'Gosund ESP 0C3994 Plug', 'Gosund ESP 1ACEE1  Socket', 'Gosund ESP 147FF9 Plug', 'Gosund ESP 10ACD8 Plug', 'HeimVision SmartLife Radio Lamp', 'Philips Hue Bridge', 'Ring Base Station AC 1236', 'iRobot Roomba', 'Smart Board', 'Teckin Plug 1', 'Teckin Plug 2', 'Yutron Plug 1', 'Yutron Plug 2', 'D-Link DCHS-161 Water Sensor', 'LG Smart TV', 'Netatmo Weather Station']

def read_pcap(path):
    packets = PcapReader(str(path))
    return packets

def shannon(data):
    LOG_BASE = 2
    # We determine the frequency of each byte
    # in the dataset and if this frequency is not null we use it for the
    # entropy calculation
    dataSize = len(data)
    ent = 0.0
    freq = {}
    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
    # to determine if each possible value of a byte is in the list
    for key in freq.keys():
        f = float(freq[key]) / dataSize
        if f > 0:  # to avoid an error for log(0)
            ent = ent + f * math.log(f, LOG_BASE)
    return -ent

def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"
    return packet

def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload
    return packet

for i, pcap_name in enumerate(files_add):
    pcap_path = "split_data/" + pcap_name
    for pck in read_pcap(pcap_path):
        if pck.src in mac_list:
            dev = dic[pck.src]
            count = dev_count[dev]
            if count > 9999:
                continue
            if pck.haslayer(ARP):
                continue
            if pck.haslayer(DHCP):
                continue
            pdata = []
            if "TCP" in pck:
                pdata = (pck[TCP].payload)
            if "Raw" in pck:
                pdata = (pck[Raw].load)
            elif "UDP" in pck:
                pdata = (pck[UDP].payload)
            elif "ICMP" in pck:
                pdata = (pck[ICMP].payload)
            pdata = list(memoryview(bytes(pdata)))

            if pdata != []:
                entropy = shannon(pdata)
            else:
                entropy = 0
            try:
                pck_size = pck.len
            except Exception as e:
                # 异常处理代码
                continue
            if entropy % 1 < 0.5:
                entropy = entropy - entropy % 1
            else:
                entropy = entropy - entropy % 1 + 1
            payload_bytes = len(pdata)
            pck = remove_ether_header(pck)
            pck = mask_ip(pck)
            arr = np.frombuffer(raw(pck), dtype=np.uint8)[0:1500]
            arr = np.insert(arr, 0, entropy)
            arr = np.insert(arr, 0, payload_bytes)
            arr = np.insert(arr, 0, pck_size)
            if len(arr) < 1503:
                # 计算需要补多少个0
                pad_width = 1503 - len(arr)
                # 对数组进行补0操作
                arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
            np.save("features/npy/" + dev + "/" + dev + "_" + str(count) + ".npy", arr)
            dev_count[dev] += 1
    print("还剩" + str(len(files_add) - i) + "个文件")




