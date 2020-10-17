import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from controller import infer

class GUI:
    import cv2
    def __init__(self, window_title, location_name, occupancy_limit, video_source = 0):
        self.window_title = window_title

        cv2.namedWindow(self.window_title, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(self.window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.camera = Camera(video_source)
        self.frame = np.zeros((self.camera.width, self.camera.height, 3))

        self.location_name = location_name
        self.occupancy_limit = occupancy_limit

        self.occupancy = 0
        self.frame_counter = 0

        #blob array is of form [((x1, y1) (x2, y2)), ...]
        self.blob_array = []

        self.pool = ThreadPool(processes=1)

        self.ml_frames = None
        self.async_result = self.pool.apply_async(self.call_model, (self.frame,))  # tuple of args for foo

        self.prev_centers = list()


    def process(self):
        self.frame = self.camera.get_frame()
        self.frame_counter += 1
        if(self.frame_counter == 9):

            self.frame_counter = 0

            # await async call
            return_val = self.async_result.get()  # get the return value

            # Box updates
            self.ml_frames = return_val

            # count update
            self.count_peeps(return_val)

            # start new async call
            self.async_result = self.pool.apply_async(self.call_model, (self.frame,))
        else:
            pass

    def count_peeps(self, model_output):

        new_centers = list()
        for output in model_output:

            x1, y1, x2, y2, label_name = output

            x_avg = (x1 + x2)/2
            y_avg = (y1 + y2)/2
            new_centers.append((x_avg, y_avg))

        self.match_centers(self.prev_centers, new_centers)
        self.prev_centers = new_centers

    def match_centers(self, old_centers, new_centers, min_dist=0.3):
        old_centers = np.array(old_centers)
        new_centers = np.array(new_centers)


        if len(new_centers) == 0 :
            return
        elif len(old_centers) == 0:
            candidate = new_centers.copy()
        else:
            dists = self.pairwise_dist(old_centers, new_centers)
            min_dists = np.amin(dists, axis=0)
            candidate = new_centers[min_dists > min_dist]


        for x, y in candidate:
            if x < 0.2:
                self.person_enter()

    def pairwise_dist(self, x, y):

        # (x-y)^2 = x^2 - 2xy + y^2

        first = np.sum(x ** 2, axis=1)
        middle = (-2) * np.dot(x, y.T)
        last = np.sum(y ** 2, axis=1)

        # For broadcasting
        dists = np.expand_dims(first, axis=1) + middle + last

        # To prevent rounding issues
        dists[dists < 0] = 0
        return dists ** (1 / 2)

    def update(self, model_output):
        # print(model_output)

        for output in model_output:

            x1, y1, x2, y2, label_name = output

            start_point = (int(x1 * self.camera.width), int(y1 * self.camera.height))
            end_point = (int(x2* self.camera.width), int(y2 * self.camera.height))
            color = (0, 0, 255)

            self.frame = cv2.rectangle(self.frame, start_point, end_point, color, 10)

    def ux_annotate(self):
        self.frame = cv2.copyMakeBorder(self.frame, int(0.1 * self.camera.height), int(0.1 * self.camera.height), 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))

        self.frame = cv2.rectangle(self.frame, (0, 0), (self.camera.width, int(0.1 * self.camera.height)), (0, 0, 0), -1)
        self.frame = cv2.rectangle(self.frame, (0, int(1.1 * self.camera.height)), (self.camera.width, int(1.2 * self.camera.height)), (0, 0, 0), -1)
        textsize = cv2.getTextSize(self.location_name, cv2.FONT_HERSHEY_COMPLEX, 1.5, 3)[0]

        self.frame = cv2.putText(self.frame, self.location_name, ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 2)

        if self.ml_frames is not None:
            self.update(self.ml_frames)

        textsize = cv2.getTextSize("Occupancy: " + str(self.occupancy), cv2.FONT_HERSHEY_COMPLEX, 1.5, 2)[0]
        if(self.occupancy < self.occupancy_limit):
            self.frame = cv2.putText(self.frame, "Occupancy: " + str(self.occupancy), ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2 + int(1.1 * self.camera.height)), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 2)
        else:
            self.frame = cv2.putText(self.frame, "Occupancy: " + str(self.occupancy), ((self.camera.width - textsize[0]) // 2, (int(0.1 * self.camera.height) + textsize[1]) // 2 + int(1.1 * self.camera.height)), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 255), 2)

    def person_annotate(self):
        if(self.occupancy < self.occupancy_limit):
            for blob in self.blob_array:
                self.frame = cv2.rectangle(self.frame, blob[0], blob[1], (0, 255, 0), 2)
        else:
            for blob in self.blob_array:
                self.frame = cv2.rectangle(self.frame, blob[0], blob[1], (0, 0, 255), 2)
        pass

    def person_enter(self):
        self.occupancy += 1

    def person_exit(self):
        self.occupancy -= 1
        if(self.occupancy <= 0):
            self.occupancy = 0

    def add_person_array(self, array):
        self.blob_array = array

    def call_model(self, image):
        return infer(image)

    def run(self):
        while True:
            self.process()
            self.person_annotate()
            self.ux_annotate()
            cv2.imshow(self.window_title, self.frame)
            if(cv2.waitKey(10) == 27 or cv2.getWindowProperty(self.window_title, 0) < 1):
                cv2.destroyAllWindows()
                break

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
