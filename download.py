#!/usr/bin/env python
# Downloads 3RScan public data release
# Run with ./download.py (or python download-scannet.py on Windows)
# -*- coding: utf-8 -*-

import sys
import argparse
import os

if sys.version_info.major >= 3 and sys.version_info.minor >= 6:
    import urllib.request as urllib
else:
    import urllib
import tempfile
import re

BASE_URL = 'http://campar.in.tum.de/files/3Dhand-RAL/'

def download_file(url, out_file):
    print(url)
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    print('\t' + url + ' > ' + out_file)
    fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
    f = os.fdopen(fh, 'w')
    f.close()
    urllib.urlretrieve(url, out_file_tmp)
    os.rename(out_file_tmp, out_file)





def main():
    parser = argparse.ArgumentParser(description='Downloads 3D Synth Grasping Hand Dataset.')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--id', help='specific scan id to download')
    parser.add_argument('--type', help='specific file type to download')
    args = parser.parse_args()
    download_file(BASE_URL + '/SynGraspHand.zip', args.out_dir + '/SynGraspHand.zip')
    print("SynGraspHand Dataset downloaded")
    

if __name__ == "__main__": main()
