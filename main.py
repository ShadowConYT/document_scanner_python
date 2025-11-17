import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

from utils import get_filename, process_image, save_as_pdf

class DocumentScannerApp:
    def __init__(self, window_title="Document Scanner"):
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.geometry("1000x800")

        self.scanned_pages = []
        self.thumbnail_images = []
        
        self.video_source = 0
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open video source {self.video_source}")
            self.window.destroy()
            return

        self.main_frame = tk.Frame(self.window)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.controls_frame = tk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, padx=10, pady=5)

        self.btn_capture = tk.Button(self.controls_frame, text="Capture Page (Space)", command=self.capture_page)
        self.btn_capture.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_save = tk.Button(self.controls_frame, text="Save PDF (Ctrl+S)", command=self.save_document)
        self.btn_save.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_clear = tk.Button(self.controls_frame, text="New/Clear", command=self.clear_document)
        self.btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_quit = tk.Button(self.controls_frame, text="Quit (Esc)", command=self.on_closing)
        self.btn_quit.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.preview_container = tk.Frame(self.main_frame, height=150)
        self.preview_container.pack(fill=tk.X, padx=10, pady=10)
        self.preview_container.pack_propagate(False)

        self.preview_canvas = tk.Canvas(self.preview_container)
        self.scrollbar = tk.Scrollbar(self.preview_container, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_frame = tk.Frame(self.preview_canvas)

        self.preview_canvas.configure(xscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.preview_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_window = self.preview_canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")

        self.preview_frame.bind("<Configure>", lambda e: self.preview_canvas.configure(scrollregion=self.preview_canvas.bbox("all")))
        self.preview_canvas.bind('<Configure>', self.on_canvas_resize)

        self.window.bind('<space>', lambda event: self.capture_page())
        self.window.bind('<Control-s>', lambda event: self.save_document())
        self.window.bind('<Escape>', lambda event: self.on_closing())

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.update_video_feed()

    def on_canvas_resize(self, event):
        self.preview_canvas.itemconfig(self.canvas_window, width=event.width)

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_raw = frame
            
            img = Image.fromarray(frame_rgb)
            
            w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
            if w > 1 and h > 1:
                img.thumbnail((w, h), Image.Resampling.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
        self.window.after(15, self.update_video_feed)

    def capture_page(self):
        if hasattr(self, 'current_frame_raw'):
            processed_img = process_image(self.current_frame_raw)
            if processed_img is None:
                processed_img = self.current_frame_raw.copy()

            self.scanned_pages.append(processed_img)
            self.add_thumbnail(processed_img)

    def add_thumbnail(self, img_np):
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)

        target_height = 120
        aspect_ratio = img.width / img.height
        new_width = int(target_height * aspect_ratio)
        img.thumbnail((new_width, target_height), Image.Resampling.LANCZOS)

        imgtk = ImageTk.PhotoImage(image=img)
        self.thumbnail_images.append(imgtk)

        thumb_label = tk.Label(self.preview_frame, image=imgtk, relief=tk.RAISED, borderwidth=2)
        thumb_label.pack(side=tk.LEFT, padx=5, pady=5)
    
    def save_document(self):
        if not self.scanned_pages:
            messagebox.showwarning("Warning", "No pages have been scanned.")
            return

        suggested_name = get_filename(self.scanned_pages[0])
        if not suggested_name: suggested_name = "document.pdf"

        filepath = filedialog.asksaveasfilename(
            initialfile=suggested_name,
            defaultextension=".pdf",
            filetypes=[("PDF Documents", "*.pdf"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                save_as_pdf(self.scanned_pages, filepath)
                messagebox.showinfo("Success", f"Document saved to {filepath}")
                self.clear_document()
            except Exception as e:
                messagebox.showerror("Error", f"Could not save PDF: {e}")

    def clear_document(self):
        self.scanned_pages.clear()
        self.thumbnail_images.clear()
        for widget in self.preview_frame.winfo_children():
            widget.destroy()

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    app = DocumentScannerApp()
    app.run()