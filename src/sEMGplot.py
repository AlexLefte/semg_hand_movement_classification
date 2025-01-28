import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt

# Funcție pentru a deschide un fișier numpy și a încărca datele
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npy")])
    if file_path:
        try:
            global data
            data = np.load(file_path)
            if len(data.shape) not in [2, 3]:  # Verificare dimensiuni (2D sau 3D)
                messagebox.showerror("Eroare", "Fișierul nu conține date valide (dimensiuni incompatibile).")
                return
            messagebox.showinfo("Succes", "Fișier încărcat cu succes!")
            
            # 1. Print max values
            # for i in range(data.shape[0]):
            #     print(f'Max channel value: {np.max(data[i])}')
            print(f'Min channel value: {np.min(data)}')
            print(f'Max channel value: {np.max(data)}')
            
        except Exception as e:
            messagebox.showerror("Eroare", f"A apărut o problemă: {e}")

# Funcție pentru a plota semnalele pe subplot-uri
def plot_signals():
    global data 
    try:
        start_time = 512*int(start_time_entry.get())
        end_time = 512*int(end_time_entry.get())
        
        if data is None:
            messagebox.showerror("Eroare", "Niciun fișier încărcat.")
            return

        if start_time < 0 or end_time > data.shape[1] or start_time >= end_time:
            messagebox.showerror("Eroare", "Interval de timp invalid.")
            return


        # Compute the sum on all channel signal
        semg = np.mean(data, axis=0)
        # data = np.vstack([data, semg])
        num_channels = data.shape[0]

        time = np.arange(start_time, end_time) / 512

        fig, axs = plt.subplots(num_channels, 1, figsize=(10, 6), sharex=True)
        fig.suptitle("Semnale EMG", fontsize=16)

        for i in range(num_channels):
            label = f'Canal {i + 1}' if i < num_channels - 1 else 'Mean sEMG'
            axs[i].plot(time, data[i, start_time:end_time], label=label)
            # axs[i].set_ylim([0, 255])
            axs[i].set_ylabel(label)
            axs[i].grid(True)
            axs[i].legend(loc="upper right", fontsize=8)

        axs[-1].set_xlabel("Timp [s]")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except Exception as e:
        messagebox.showerror("Eroare", f"A apărut o problemă: {e}")

# Crearea interfeței grafice
root = tk.Tk()
root.title("Analiză Semnale EMG")

data = None  # Variabilă globală pentru stocarea datelor

# Butoane și câmpuri de introducere
open_button = tk.Button(root, text="Încarcă fișier .npy", command=open_file)
open_button.pack(pady=10)

start_time_label = tk.Label(root, text="Timp de start:")
start_time_label.pack()
start_time_entry = tk.Entry(root)
start_time_entry.pack(pady=5)

end_time_label = tk.Label(root, text="Timp de sfârșit:")
end_time_label.pack()
end_time_entry = tk.Entry(root)
end_time_entry.pack(pady=5)

plot_button = tk.Button(root, text="Plotează semnalele", command=plot_signals)
plot_button.pack(pady=10)

# Pornirea buclei principale
root.mainloop()
