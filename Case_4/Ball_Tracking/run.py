import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Detectors(object):

    def __init__ (self):

        #To create the object for backgrounf subtraction from individual frames
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    def detectBall(self, img):

        #Convert RGB to Gray and apply background subtraction
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fgmask = self.fgbg.apply(gray)

        # Detect all contours in the image
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #Find valid contours (radius greater than 10) and their minenclosing circles
        circles = []
        validContours = []
        for contour in contours:
            try:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if(radius > 10):
                    circles.append([x,y,radius])
                    validContours.append(contour)
            except ZeroDivisionError:
                pass

        #Find convex hull of sets of neighbouring contours
        l = len(circles)
        sets = np.arange(1,l+1,1)
        circles = np.array(circles)
        validContours = np.array(validContours)
        thresh = 25   #Distance threshold
        for i in range(0,l-1,1):
            for j in range(i+1, l,1):
                dist = np.linalg.norm(circles[i][0:2]-circles[j][0:2])
                if(dist < thresh):
                    sets[j] = sets[i]

        convHull = []
        for i in np.unique(sets):
            idx = np.where(sets == i)
            cont = np.vstack(validContours[i] for i in idx[0])
            convHull.append(cv2.convexHull(cont))

        similarity = []
        centers = []
        for ch in convHull:
            try:
                (x,y), radius = cv2.minEnclosingCircle(ch)
                theta = np.arange(0, 2*np.pi, 2*np.pi/len(ch))
                theta = theta.reshape(len(theta),1)
                xa = np.int32(x+ radius*np.cos(theta))
                ya = np.int32(y+ radius*np.sin(theta))
                circle = np.array([xa.T,ya.T]).T
                centers.append([x,y,radius])
                similarity.append(cv2.matchShapes(ch, circle,1,0.0)/len(ch))
            except ZeroDivisionError:
                pass

        #find the most similar convex hull and min enclosing circles
        try:
            index = np.argsort(similarity)[0]
            #cv2.circle(img, (int(centers[index][0]), int(centers[index][1])), int(centers[index][2]), (255,255,255), 3)
            #plt.imshow(img)
            #plt.show()
            return centers[index]
        except IndexError:
            return None

class KalmanFilter(object):

    def __init__(self,bx,by, framerate, skipframes):

        # delta T - time difference
        self.dt = skipframes*framerate

        # Previous state vector {x,y,vx,vy, ax, ay}
        self.xkp = np.array([bx, by, 0, 0, -9.8, 0])
        self.xk = self.xkp

        # Measured vector
        self.xm = np.array([0, 0])

        #State Transition Matrix
        self.F = np.eye(self.xkp.shape[0])
        self.F[0][2] = self.dt
        self.F[0][4] = (self.dt**2)/2
        self.F[1][3] = self.dt
        self.F[1][5] = (self.dt**2)/2
        self.F[2][4] = self.dt
        self.F[3][5] = self.dt

        # Initial Process Covariance Matrix
        self.Pkp = np.eye(self.xkp.shape[0])

        # Process Noise Covariance Matrix
        self.Qk = 100*np.eye(self.xkp.shape[0])

        # Control Matrix
        #self.Bk = np.array([(self.dt**2)/2, self.dt])

        #Control Vector - initialised to acceleration due to gravity
        #self.uk = np.array([-9.8])

        #Sensor Matrix
        self.Hk = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])

        #Measurement covariance matrix
        self.R = 10*np.eye(self.xm.shape[0])

    def predict(self):

        #Predicted Vector
        self.xk = self.F @ self.xkp #+ self.Bk @ self.uk

        #Setting Previously predicted to current for next frame
        self.xkp = self.xk

        #Getting the updated process cov matrix
        self.Pkp = self.F @ self.Pkp @ self.F.T

        return self.xk

    def update(self,bx,by):

        #Update Measurement Vector
        self.xm = np.array([bx,by])
        #Kalman Gain
        A = self.Hk @ self.Pkp @ self.Hk.T + self.R
        K = np.linalg.inv(A)
        K = self.Hk.T @ K
        K = self.Pkp @ K

        #Most Likely state Vector
        self.xk = self.xk + K @ (self.xm - self.Hk @ self.xk)

        #Updated Process covariance Matrix
        self.Pkp = self.Pkp - K @ self.Hk @ self.Pkp

        return self.xk

### Run

def main():

    start_time = time.time()

    vid = cv2.VideoCapture('Case_4/banh_phuoc.mp4')

    fps = vid.get(cv2.CAP_PROP_FPS)
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))

    detector = Detectors()

    skipframes = 1 #No of frames skipped + 1 (Time between successive frames given for detection)
    flag = 1 #To find the first detection

    track = [] #Keeps Track of most likely state vectors
    frames = [] #Stores frames of the video

    while(vid.isOpened()):
        ret, frame = vid.read()
        if ret == True:
            frames.append(frame)
            bcenter = detector.detectBall(frame)
            try:
                if(len(bcenter) > 0 and flag == 1):
                    KF = KalmanFilter(bcenter[0], bcenter[1], fps, skipframes)
                    track.append(KF.xk)
                    flag = 0

                xp = KF.predict()
                xmp = KF.update(bcenter[0], bcenter[1])
                track.append(xmp)
            except TypeError:
                pass
        else:
            break

    vid.release()

    print("Time to execute : %s seconds" % (time.time()-start_time))

    #Writing to video
    out = cv2.VideoWriter('vid2Kalman.avi',cv2.VideoWriter_fourcc('M','P','E','G'), 10, (frame_width,frame_height))
    for i in range(len(frames)):
      print(i)
      cv2.circle(frames[i], (int(track[i][0]), int(track[i][1])), 10, (255,255,255), 4)
      out.write(frames[i])

    out.release()

if __name__ == "__main__":
    main()
