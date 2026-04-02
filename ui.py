import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading

from engine import scan_files


root = tk.Tk()
root.title("Audio Analyzer")
root.geometry("750x550")

folder_var = tk.StringVar(value=r"N:\Music")
scan_subfolders_var = tk.BooleanVar()
force_refresh_var = tk.BooleanVar()


log_box = None


def log(msg):
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)
    root.update_idletasks()


def browse():
    folder = filedialog.askdirectory()
    folder_var.set(folder)


def run_scan():
    scan_files(
        folder_var.get(),
        scan_subfolders_var.get(),
        force_refresh_var.get(),
        log
    )


def start_scan():
    threading.Thread(target=run_scan, daemon=True).start()


frame = tk.Frame(root)
frame.pack(pady=10)

tk.Label(frame, text="Folder:").grid(row=0, column=0)

tk.Entry(frame, textvariable=folder_var, width=55).grid(row=1, column=0)
tk.Button(frame, text="Browse", command=browse).grid(row=1, column=1)


tk.Checkbutton(root, text="Scan subfolders", variable=scan_subfolders_var).pack(anchor="w", padx=10)
tk.Checkbutton(root, text="Force refresh", variable=force_refresh_var).pack(anchor="w", padx=10)

tk.Button(root, text="START SCAN", command=start_scan, bg="green", fg="white").pack(pady=10)

log_box = scrolledtext.ScrolledText(root, height=22)
log_box.pack(fill="both", expand=True, padx=10, pady=10)

