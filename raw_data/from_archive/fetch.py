import os, sys, shutil
import re

dirs = ["//suzhost-17/training_archive/official/WP/chs_alarm/2014_01_07_01h_46m_43s",
        "//stcsrv-g31/training_archive/official/WP/chs_calendar/2014_01_05_23h_15m_39s",
        "//stcsrv-g31/training_archive/official/WP/chs_Communication/2013_12_31_22h_45m_59s",
        "//stcsrv-g31/training_archive/official/WP/chs_note/2013_12_30_21h_36m_45s",
        "//stcsrv-g31/training_archive/official/WP/chs_places/2013_11_29_19h_06m_33s",
        "//stcsrv-g31/training_archive/official/WP/chs_reminder/2013_12_30_23h_22m_43s",
        "//stcsrv-g31/training_archive/official/WP/chs_weather/2013_12_26_01h_16m_50s",
        "//stcsrv-g31/training_archive/official/WP/chs_web/2013_11_19_01h_31m_40s"]

output_prefix = "/home/juluan/tmp/cortana_data/train/zh-cn"

def fetch_dir(path):
    data_path = "%s/wordbroken/" % path
    domain = re.search(r".+/chs_(.+?)/", path).group(1).lower()
    files = os.listdir(data_path)
    outpath = "%s/%s" % (output_prefix, domain)
    if not os.path.exists(outpath):
        os.makedirs(outpath, 0755)

    print "fetching %d data files for %s, destination: %s" % \
        (len(files), domain, outpath)

    for filename in files:
        print "copying %s" % filename
        shutil.copy2("%s/%s" % (data_path, filename), outpath)

for directory in dirs:
    fetch_dir(directory)
    
