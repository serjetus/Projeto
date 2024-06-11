import threading
import time
from thread import WhatsAppThread
import os


class MessageThread(threading.Thread):
    def __init__(self, telefone, image_path, message, tipe):
        threading.Thread.__init__(self)
        self.tipe = tipe
        self.telefone = telefone
        self.image_path = image_path
        self.message = message

    def run(self):
        time.sleep(5)
        whatsapp_thread = WhatsAppThread(self.telefone, self.image_path, self.message, self.tipe)
        whatsapp_thread.start()
