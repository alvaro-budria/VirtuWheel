# TODO: WARNING TO MALIKA FROM ALVARO:
#  remember to lower the threshold for the visibilities of the wrists that is hardcoded in the library lib/python3.9/site-packages/mediapipe/python/solutions/drawing_utils.py
import cv2
import numpy as np
import mediapipe as mp


# a small hack to avoid crashing when the wrists are not detected
LAST_ANGLE = 0
LAST_POSE_LANDMARKS = None
IT_USED_LAST_POSE_LANDMARKS = 0


def compute_left_right(points):
    """
    Grabs the lefth and right wrist coordinates.
    Assumes that the points define a diameter of a circle, whose center is at the midpoint of the diameter.
    It then computes the angle of the diameter with respect to the horizontal axis.
    """
    global LAST_ANGLE

    # Left wrist
    left_wrist = points[15]
    left_wrist_vis = left_wrist.visibility
    left_wrist = [left_wrist.x, left_wrist.y]
    # Right wrist
    right_wrist = points[16]
    right_wrist_vis = right_wrist.visibility
    right_wrist = [right_wrist.x, right_wrist.y]

    # Compute the angle of the diameter with respect to the horizontal axis
    # The diameter is defined by the left and right wrists
    # The center of the diameter is at the midpoint of the diameter
    # The angle is computed using the arctangent function
    # The angle is in radians, so we convert it to degrees
    # The angle is in the range [-90, 90]
    # The angle is 0 if the diameter is horizontal
    # The angle is 90 if the diameter is vertical
    # The angle is positive if the diameter is above the horizontal axis
    # The angle is negative if the diameter is below the horizontal axis
    # here we set high thresholds for the visibilities because we want to make sure that the wrists are detected
    # and because only when both arms are detected, does MediaPipe return high visibilities
    if left_wrist is None or right_wrist is None or left_wrist_vis < 0.75 or right_wrist_vis < 0.75 or \
        np.abs(left_wrist_vis - right_wrist_vis) > 0.15:
        # this is a hack to avoid crashing when the wrists are not detected
        return LAST_ANGLE
    angle = np.degrees(np.arctan2(left_wrist[1] - right_wrist[1], left_wrist[0] - right_wrist[0]))
    # clip angle to be in the range [-90, 90]
    angle = np.clip(angle, -90, 90)
    # normalize angle to be in the range [-1, 1]
    angle = angle / 90
    LAST_ANGLE = angle
    return angle


def main():
    global LAST_POSE_LANDMARKS
    global IT_USED_LAST_POSE_LANDMARKS

    # Initializing the MediaPipe Pose and Drawing modules.
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # define a video capture object
    vid = cv2.VideoCapture(0)

    with mp_pose.Pose(static_image_mode=True) as pose:
        while(True):
            # Capture the video frame by frame
            ret, frame = vid.read()

            frameHeight = frame.shape[0]

            # This function uses the MediaPipe library to detect and draw landmarks in an image.
            # The landmarks are points of interest that represent different body parts detected

            # Makes a copy of the original image.
            annotated_img = frame.copy()

            # Process the image.
            results = pose.process(frame)

            # Define the radius of the circle for drawing the 'landmarks'.
            # The radius is scaled as a percentage of the image height.
            circle_radius = int(.01 * frameHeight)

            # Specifies the drawing style of the 'landmarks'.
            point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-1, circle_radius=circle_radius)

            if results.pose_landmarks is not None:
                # Draws the 'landmarks' in the image.
                list_landmarks = [landmark for landmark in results.pose_landmarks.landmark]
                angle = compute_left_right(list_landmarks)
                # clip angle to be in range [-1, 0.1] U [0.1, 1]
                angle = np.clip(angle, -1, -0.15) + np.clip(angle, 0.15, 1)
                angle *= -1.0
                if angle is not None:
                    mp_drawing.draw_landmarks(
                        annotated_img, landmark_list=results.pose_landmarks, landmark_drawing_spec=point_spec
                    )
                    LAST_POSE_LANDMARKS = results.pose_landmarks
                    IT_USED_LAST_POSE_LANDMARKS = 0
                    with open('angle_input.txt', 'w') as file:
                        file.write(str(angle))
            elif IT_USED_LAST_POSE_LANDMARKS > 11:  # do not use the last pose landmarks more than 11 times
                mp_drawing.draw_landmarks(
                    annotated_img, landmark_list=results.pose_landmarks, landmark_drawing_spec=point_spec
                )
            else:
                mp_drawing.draw_landmarks(
                    annotated_img, landmark_list=LAST_POSE_LANDMARKS, landmark_drawing_spec=point_spec
                )
                IT_USED_LAST_POSE_LANDMARKS += 1

            cv2.imshow('Output-Keypoints', annotated_img)

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