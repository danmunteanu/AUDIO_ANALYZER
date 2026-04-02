from ui import root

if __name__ == "__main__":
    # This is the ONLY place mainloop() should be called.
    # It ensures the UI opens once, and workers stay silent.
    root.mainloop()