import threading
import time

import pywhatkit as kt
import os


class WhatsAppThread(threading.Thread):
    def __init__(self, telefone, image_path, message, tipe):
        threading.Thread.__init__(self)
        self.tipe = tipe
        self.telefone = telefone
        self.image_path = image_path
        self.message = message

    def run(self):
        time.sleep(3)
        if self.tipe == 1:
            kt.sendwhats_image(self.telefone, self.image_path, self.message)
        else:
            kt.sendwhatmsg_instantly(self.telefone, self.message)
