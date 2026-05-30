import cv2
import py360convert
import numpy as np
import progressbar
import argparse
import os
import csv
import random
from multiprocessing import Pool


class input_data(object):
    def __init__(self, filename, elevation):
        self.f_name = filename
        self.elevation = elevation


def readDataset(fname):
    data_list = []
    with open(fname, newline='') as csvfile:
        rows = csv.DictReader(csvfile)
        for row in rows:
            data_list.append(input_data(
                row['image_path'],
                row['elevation_deg']
            ))
    return data_list



def main(args):

    data_list = readDataset(args.input_csv)

    print("doing the work")
    pano_list = []
    output_list = []
    for i in progressbar.progressbar(range(len(data_list))):
        pano = cv2.imread(data_list[i].f_name)
        pano = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
        
        num_samples = 12

        for _ in range(num_samples):
            yaw = random.uniform(0, 360)
            fov = random.uniform(50, 90)
            vert = random.uniform(-30, 30)

            img = py360convert.e2p(
                pano,
                fov_deg=fov,
                u_deg=yaw,
                v_deg=vert,
                out_hw=(256,256)
            )
            filename = f"{i}-{yaw}-{fov}-{vert}-{data_list[i].elevation}.png"
            output_list.append({
                'filename': filename,
                'elevation': data_list[i].elevation,
                'FOV': fov,
                'v_deg': vert
            })
            cv2.imwrite(f"{args.output_img_dir}/{filename}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
    print("done doing work!")

    print("making csv")
    with open(args.output_csv, 'w', newline='') as f:
        fieldnames = ["filename", "elevation", "FOV", "v_deg"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_list)

    print("DONE")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--output_img_dir', required=True)
    args = parser.parse_args()
    main(args)
    
    