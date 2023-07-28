import os
import shutil
import glob
import cv2


BaseDir="./labeled-videos-Processed/"
for foldername in  glob.glob(BaseDir+"Frames/*"):
    videoName=os.path.splitext(os.path.basename(foldername))[0]
    NewDir=BaseDir+"Resized/Frames/"+videoName

    try:
        # creating a folder named data
        if not os.path.exists(NewDir):
            os.makedirs(NewDir)

    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of Frames')
print("done creating directories")

for foldername in  glob.glob(BaseDir+"Frames/*"):
    print(foldername)
    for filename in  glob.glob(foldername+"/*"):
        # print(filename)
        # reading the extracted images
        img=cv2.imread(filename)
        height, width, channels = img.shape
        sides = height if height < width else width
        hrem=height-sides
        wrem=width-sides
    
        # Resizing the images
        imCrop=img[int(hrem/2):int(height-hrem/2),int(wrem/2):int(width-wrem/2),:]
        imResize=cv2.resize(imCrop, (576,576))
        Nsiz=576-448
        im448=imResize[int(Nsiz/2):int(576-Nsiz/2),int(Nsiz/2):int(576-Nsiz/2),:]
        imRes=cv2.resize(im448, (288,288))
        # writing images to new folder
        ResizeName=filename.split("Frames/")[1]
        ResizeDir=BaseDir+"Resized/Frames/"+ResizeName
        # print(ResizeDir)
        cv2.imwrite(ResizeDir,imRes)

    cv2.destroyAllWindows()
