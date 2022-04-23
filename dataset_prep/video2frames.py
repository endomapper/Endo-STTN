import os
import shutil 
import glob
import cv2


BaseDir="./labeled-videos-Processed/"
for filename in  glob.glob("./labeled-videos-Processed/Videos/*"):
    videoName=os.path.splitext(os.path.basename(filename))[0]
    NewDir=BaseDir+"Frames/"+videoName
    # Read the video from specified path 
    cam = cv2.VideoCapture(filename) 

    try: 
        # creating a folder named data 
        if not os.path.exists(NewDir): 
            os.makedirs(NewDir) 

    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of Frames') 

    # frame 
    currentframe = 0

    while(True): 

        # reading from frame 
        ret,frame = cam.read() 

        if ret: 
            # if video is still left continue creating images 
            name = NewDir + "/{:05}".format(currentframe)+".jpg"
            print ('Creating...' + name) 

            # writing the extracted images 
            cv2.imwrite(name, frame) 

            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1
        else: 
            break

    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()

