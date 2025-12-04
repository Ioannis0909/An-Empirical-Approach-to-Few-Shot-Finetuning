#!/usr/bin/env bash
# wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
# tar -zxvf CUB_200_2011.tgz
# python write_CUB_filelist.py

#!/usr/bin/env bash
set -e

# remove any previous broken HTML "tgz" file
rm -f CUB_200_2011.tgz

# working mirror
wget https://media.githubusercontent.com/media/vignagajan/CUB-200-2011/main/CUB_200_2011.tgz

tar -zxvf CUB_200_2011.tgz
python write_CUB_filelist.py

'''
cd ~/Desktop/Bird_Models/CloserLookFewShot/filelists/CUB
chmod +x download_CUB.sh
./download_CUB.sh
'''