import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox
from PIL import Image, ImageTk
import datetime
import re
from main import process


class VideoApp:
    def __init__(self, root):
        self.warning_count = None
        self.detection_threshold = "2"
        self.time_threshold = "10"
        self.time_thresholdP = "5"
        self.pixels = None
        self.fps = None
        self.root = root
        self.root.title("Visão da camera")

        self.paused = False
        self.cap = None
        self.roi_count = 0
        self.frameCount = 0
        self.rois = []


        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        self.play_button = ttk.Button(root, text="Play | Pause", command=self.toggle_play_pause)
        self.play_button.pack(side=tk.LEFT)

        self.select_roi_button1 = ttk.Button(root, text="ROI Garagem", command=lambda: self.select_roi(1))
        self.select_roi_button1.pack(side=tk.LEFT)

        self.select_roi_button2 = ttk.Button(root, text="ROI Portão", command=lambda: self.select_roi(2))
        self.select_roi_button2.pack(side=tk.LEFT)

        self.select_roi_button3 = ttk.Button(root, text="ROI Calçada", command=lambda: self.select_roi(3))
        self.select_roi_button3.pack(side=tk.LEFT)

        self.load_button = ttk.Button(root, text="Carregar Video", command=self.load_video)
        self.load_button.pack(side=tk.LEFT)

        self.settings_button = ttk.Button(root, text="Configurações", command=self.open_settings_window)
        self.settings_button.pack(side=tk.LEFT)

        self.update_frame()

    def load_video(self):
        if (self.warning_count == None):
            tkinter.messagebox.showwarning("Telefone não informado", "Informe o telefone nas configuraçoes")
        else:
            video_path = filedialog.askopenfilename()
            if video_path:
                self.cap = cv2.VideoCapture(video_path)
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.pixels = int((24 / self.fps) * 15)
                ret, frame = self.cap.read()
                if ret:
                    frame_height, frame_width = frame.shape[:2]

                    scale_factor = min(640 / frame_width, 480 / frame_height, 1)
                    self.resized_width = int(frame_width * scale_factor)
                    self.resized_height = int(frame_height * scale_factor)

                    self.canvas.config(width=self.resized_width, height=self.resized_height)

    def toggle_play_pause(self):
        self.paused = not self.paused

    def select_roi(self, roi_num):
        if self.cap and not self.paused:
            self.paused = True
            ret, frame = self.cap.read()
            if ret:
                roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = roi
                roi_cropped = frame[int(y):int(y + h), int(x):int(x + w)]
                self.rois.append((roi_num, roi_cropped))
                cv2.destroyWindow("Select ROI")
                self.paused = False

    def open_settings_window(self):
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Configurações")

        self.warning_count_label = ttk.Label(settings_window, text="Telefone de aviso:")
        self.warning_count_label.pack(side=tk.LEFT)
        self.warning_count_var = tk.StringVar(value=self.warning_count or "")
        self.warning_count_entry = ttk.Entry(settings_window, textvariable=self.warning_count_var)
        self.warning_count_entry.pack(side=tk.LEFT)

        self.detection_threshold_label = ttk.Label(settings_window, text="Limite de Detecções:")
        self.detection_threshold_label.pack(side=tk.LEFT)
        self.detection_threshold_var = tk.StringVar(value=self.detection_threshold)
        self.detection_threshold_entry = ttk.Entry(settings_window, textvariable=self.detection_threshold_var)
        self.detection_threshold_entry.pack(side=tk.LEFT)

        self.time_threshold_label = ttk.Label(settings_window, text="Tempo Carro:")
        self.time_threshold_label.pack(side=tk.LEFT)
        self.time_threshold_var = tk.StringVar(value=self.time_threshold)
        self.time_threshold_entry = ttk.Entry(settings_window, textvariable=self.time_threshold_var)
        self.time_threshold_entry.pack(side=tk.LEFT)

        self.time_thresholdP_label = ttk.Label(settings_window, text="Tempo Pessoa:")
        self.time_thresholdP_label.pack(side=tk.LEFT)
        self.time_thresholdP_var = tk.StringVar(value=self.time_thresholdP)
        self.time_thresholdP_entry = ttk.Entry(settings_window, textvariable=self.time_thresholdP_var)
        self.time_thresholdP_entry.pack(side=tk.LEFT)

        self.apply_button = ttk.Button(settings_window, text="APLICAR", command=self.apply_settings)
        self.apply_button.pack(side=tk.LEFT)

    def apply_settings(self):
        try:
            if self.warning_count_var.get() and self.detection_threshold_var.get() and self.time_threshold_var.get() and self.time_thresholdP_var.get():
                self.warning_count = self.warning_count_var.get()
                if not re.match(r'^\d{11}$', self.warning_count):
                    tkinter.messagebox.showwarning("Formato Inválido",
                                                   "O número de telefone deve conter 11 dígitos.")
                    return False
                else:
                    self.warning_count = "+55" + self.warning_count
                    self.detection_threshold = int(self.detection_threshold_var.get())
                    self.time_threshold = int(self.time_threshold_var.get())
                    self.time_thresholdP = int(self.time_thresholdP_var.get())
                    print(
                        f"Settings Applied: Telefone = {self.warning_count}, Detecçoes Minimas = {self.detection_threshold}, Tempo Carro = {self.time_threshold}, Tempo Pessoa = {self.time_thresholdP}")
                    return True
            else:
                tkinter.messagebox.showwarning("Campo Vazio", "Um ou mais campos de entrada estão vazios.")
                return False
        except ValueError as e:
            tkinter.messagebox.showwarning("Erro", f"Erro: {e}")
            return False

    def update_frame(self):
        if self.cap and not self.paused:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 480))
            process(frame, self.frameCount, self.fps, self.pixels, self.time_threshold, self.warning_count)
            self.frameCount = self.frameCount + 1
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.resized_width, self.resized_height))
                frame = Image.fromarray(frame)
                frame = ImageTk.PhotoImage(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame)
                self.canvas.image = frame

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoApp(root)
    root.mainloop()
