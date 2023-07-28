import shutil
import glob

i=0
for filename in  glob.glob("./labeled-videos/*/*/*/*"):
    print(i)
    Newfilename="hyperK_{:03}".format(i)+".avi"
    shutil.copyfile(filename,"./labeled-videos-Processed/Videos/"+Newfilename,)
    i+=1

