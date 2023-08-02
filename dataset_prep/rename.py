import shutil
import glob
import os

i=0
os.makedirs("./labeled-videos-Processed/Videos", exist_ok=True)
for filename in  glob.glob("./labeled-videos/*/*/*/*"):
    print(i)
    Newfilename="hyperK_{:03}".format(i)+".avi"
    shutil.copyfile(filename,"./labeled-videos-Processed/Videos/"+Newfilename,)
    i+=1

