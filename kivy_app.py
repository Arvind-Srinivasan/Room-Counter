# from kivy.app import App
# from kivy.lang import Builder
# from kivy.uix.screenmanager import ScreenManager, Screen
# from kivy.properties import ObjectProperty
# from kivy.uix.popup import Popup
# from kivy.uix.label import Label
#
# from gui import GUI
#
#
# class MainWindow(Screen):
#     pass
#
#
# class WindowManager(ScreenManager):
#     pass
#
#
# kv = Builder.load_file("my.kv")
#
# sm = WindowManager()
#
# screens = [MainWindow(name="main")]
# for screen in screens:
#     sm.add_widget(screen)
#
# sm.current = "main"
#
# from kivy.uix.button import Button
#
# class MyMainApp(App):
#     def build(self):
#         return sm
#
# import multiprocessing
#
# import threading
#
# def run_main_thread():
#     MyMainApp().run()
#
# def run_camera_thread():
#     gui = GUI("Occupancy Tracking", "Clough Undergraduate Learning Commons", 15)
#     gui.run()
#
# import multiprocessing
# import subprocess
#
# if __name__ == "__main__":
#     from kivy.core.window import Window
#
#     # Window.fullscreen = True
#     from subprocess import Popen as start
#
#     process = start(['python', './gui.py'])
#
#     MyMainApp().run()
#
#
#
#     # camera_thread = threading.Thread(target=run_camera_thread())
#     # main_thread = threading.Thread(target=run_main_thread())
#     #
#     # camera_thread.run()
#     # main_thread.run()
#
#
#
#
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time
Builder.load_string('''
<CameraClick>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (1280, 960)
        play: False
    ToggleButton:
        text: 'Play'
        on_press: camera.play = not camera.play
        size_hint_y: None
        height: '48dp'
    Button:
        text: 'Capture'
        size_hint_y: None
        height: '48dp'
        on_press: root.capture()
''')


class CameraClick(BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")


class TestCamera(App):

    def build(self):
        return CameraClick()

if __name__ == '__main__':
    TestCamera().run()