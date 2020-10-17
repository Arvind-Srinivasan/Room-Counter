import tkinter
import cv2
from PIL import ImageTk
import PIL

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source
        self.vid = MyVideoCapture(video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 1

        self.update()

        self.window.mainloop()


    def update(self):
        # Get a frame from the video source

        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))

        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.window.after(self.delay, self.update)

class MyVideoCapture:
    def __init__(self, video_source=0):
         # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

   # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


    def get_frame(self):
        ret = None
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
if __name__ == '__main__':
    App(tkinter.Tk(), "Tkinter and OpenCV")
