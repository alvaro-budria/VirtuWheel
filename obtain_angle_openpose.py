import cv2
import time
import numpy as np


# a small hack to avoid crashing when the wrists are not detected
LAST_ANGLE = 0
LAST_POSE_LANDMARKS = None


def compute_left_right(points, probs):
    """
    Grabs the lefth and right wrist coordinates.
    Assumes that the points define a diameter of a circle, whose center is at the midpoint of the diameter.
    It then computes the angle of the diameter with respect to the horizontal axis.
    """

    # Left wrist
    left_wrist = points[4]
    # Right wrist
    right_wrist = points[7]

    global LAST_ANGLE
    if left_wrist is None or right_wrist is None:# or left_prob < 0.75 or right_prob < 0.75 or \
        # np.abs(left_prob - right_prob) > 0.15:
        # this is a hack to avoid crashing when the wrists are not detected
        # print('LAST_ANGLE ', LAST_ANGLE)
        return LAST_ANGLE

    angle = np.degrees(np.arctan2(right_wrist[1] - left_wrist[1], right_wrist[0] - left_wrist[0]))
    # print('before angle ', angle)
    # clip angle to be in the range [-90, 90]
    angle = np.clip(angle, -90, 90)
    # normalize angle to be in the range [-1, 1]
    angle = angle / 90
    # print('after angle ', angle)
    angle = np.clip(angle, -1, -0.075) + np.clip(angle, 0.075, 1)
    LAST_ANGLE = angle
    return angle


######## OpenPose
MODE = "COCO"

if MODE == "COCO":
    protoFile = "./OpenPose_models/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "./OpenPose_models/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
elif MODE == "MPI" :
    protoFile = "./OpenPose_models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./OpenPose_models/pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
########

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Specify the input image dimensions
inWidth = 96
inHeight = 96

t = time.time()

def main():
    global LAST_POSE_LANDMARKS

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while(True):
        # Capture the video frame by frame
        ret, frame = vid.read()
        frameCopy = np.copy(frame)

        t = time.time()

        # Prepare the frame to be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

        # Set the prepared object as the input blob of the network
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        threshold = 0.1
        # Empty list to store the detected keypoints
        points = []
        probs = []

        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold :
                cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
                probs.append(prob)
            else :
                points.append(None)
                probs.append(None)

        angle = compute_left_right(points, probs)
        angle = - angle
        # print('angle ', angle)
        with open('angle_input.txt', 'w') as file:
            file.write(str(angle))

        cv2.imshow('Output-Keypoints', frameCopy)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows 
    cv2.destroyAllWindows()
    cv2.waitKey(0)


if __name__ == '__main__':
    main()