from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import ObjectProperty
import cv2

from scripts.inference import Inference


transform_flag = False

# 撮影ボタン
class ImageButton(ButtonBehavior, Image):
    preview = ObjectProperty(None)

    # ボタンを押したときに実行
    def on_press(self):
        global transform_flag
        transform_flag = not transform_flag

class CameraPreview(Image):
    def __init__(self, **kwargs):
        super(CameraPreview, self).__init__(**kwargs)
        # 0番目のカメラに接続
        self.capture = cv2.VideoCapture(0)
        # 描画のインターバルを設定
        Clock.schedule_interval(self.update, 1.0 / 30)
        # pixel2style2pixelを初期化
        self.inference = Inference()

    # インターバルで実行する描画メソッド
    def update(self, dt):
        # フレームを読み込み
        ret, frame = self.capture.read()
        # pixel2style2pixelで変換
        if transform_flag:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.inference.run(frame)
        # Kivy Textureに変換
        buf = cv2.flip(frame, 0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # インスタンスのtextureを変更
        self.texture = texture

class MainScreen(Widget):
    pass

class MyCameraApp(App):
    def build(self):
        return MainScreen()

if __name__ == '__main__':
    MyCameraApp().run()