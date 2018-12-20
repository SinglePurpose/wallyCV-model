# Convert labeled xmls for each image into a single csv file.
# Run this script in your 'dataset' folder with 'eval' and 'train' sub-folders.
# This script provides csv with COMMA as delimiter.

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_ready = pd.DataFrame(xml_list, columns=column_name)
    return xml_ready


def main():
    for folder in ['train', 'eval']:
        image_path = os.path.join(os.getcwd(), ('dataset/dataset/' + folder))
        xml_temp = xml_to_csv(image_path)
        xml_temp.to_csv(('~/Desktop/' + folder + '_labels.csv'), index=None)
        print('Jobs\'s done!')


if __name__ == '__main__':
    main()
