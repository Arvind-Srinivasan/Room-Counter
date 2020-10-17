import cv2
import numpy as np

class GUI:
    def __init__(self, window_title, location_name, occupancy_limit, video_source = 0):
        self.window_title = window_title

        cv2.namedWindow(self.window_title)

        self.camera = Camera(video_source)
        self.frame = np.zeros((self.camera.width, self.camera.height, 3))

        self.location_name = location_name
        self.occupancy_limit = occupancy_limit

        self.occupancy = 0

        self.frame_counter = 0

    def __del__(self):
        cv2.destroyWindow(self.window_title)

    def process(self):
        self.frame = self.camera.get_frame()
        self.frame_counter += 1
        if(self.frame_counter == 9):
            #shit goes here
            self.frame_counter = 0;
            pass
        else:
            pass

    def ux_annotate(self):
        self.frame = cv2.rectangle(self.frame, (0, 0), (self.camera.width, int(0.1 * self.camera.height)), (0, 0, 0), -1)
        self.frame = cv2.rectangle(self.frame, (0, self.camera.height), (self.camera.width, int(0.9 * self.camera.height)), (0, 0, 0), -1)
        textsize = cv2.getTextSize(self.location_name, cv2.FONT_HERSHEY_COMPLEX, 1.5, 3)[0]

        self.frame = cv2.putText(self.frame, self.location_name, ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)

        textsize = cv2.getTextSize("Occupancy: " + str(self.occupancy), cv2.FONT_HERSHEY_COMPLEX, 1.5, 3)[0]
        if(self.occupancy < self.occupancy_limit):
            self.frame = cv2.putText(self.frame, "Occupancy: " + str(self.occupancy), ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2 + int(0.9 * self.camera.height)), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 3)
        else:
            self.frame = cv2.putText(self.frame, "Occupancy: " + str(self.occupancy), ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2 + int(0.9 * self.camera.height)), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 3)

    #blob array is of form [((x1, y1) (x2, y2)), ...]
    def person_annotate(self, blob_array):
        pass

    def person_enter(self):
        self.occupancy += 1;

    def person_exit(self):
        self.occupancy -= 1;
        if(self.occupancy <= 0):
            self.occupancy = 0;

    def run(self):
        while True:
            self.process()
            self.ux_annotate()
            self.person_annotate([((0, 0), (0, 0))])
            cv2.imshow(self.window_title, self.frame)
            cv2.waitKey(1)

class Camera:
    def __init__(self, video_source):
        self.camera = cv2.VideoCapture(video_source)

        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __del__(self):
        if self.camera.isOpened():
            self.camera.release()

    def get_frame(self):
        if self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                return frame
        return None

if __name__ == '__main__':
    gui = GUI("Occupancy Tracking", "Clough Undergraduate Learning Commons", 15)
    gui.run()
