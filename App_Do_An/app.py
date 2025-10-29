import tkinter as tk
import customtkinter as ctk
try:
    import darkdetect
except:
    pass

# Local
from settings import *
from signin import LoginScreen
from menu import Menu
from vidframe import VidFrame
from statsframe import StatsTabView 
from sqlite import BottleDB



class App(ctk.CTk):
    def __init__(self, is_dark):
        super().__init__()

        # setup
        self.title("Đồ Án")
        self.geometry("1400x700")
        ctk.set_appearance_mode(f'{"dark" if is_dark else "light"}')
        try:
            self.iconbitmap("water-bottle.ico")
        except:
            pass
        self.minsize(1400, 700)
        
        # Hide main window initially
        self.withdraw()
        
        # Show login screen
        self.show_login()

    def show_login(self):
        """Show login screen"""
        login_screen = LoginScreen(self, self.on_login_success)
        
    def on_login_success(self):
        """Called when login is successful"""
        # Show main window
        self.deiconify()
        
        # Setup main app layout
        self.setup_main_app()
        
    def setup_main_app(self):
        """Setup the main application after successful login"""
        # layout
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=7, uniform='a')  # Video 
        self.columnconfigure(1, weight=3, uniform='a')  # Stats 

        # Menu
        self.menu = Menu(self)
        self.config(menu=self.menu)

        # Video source
        video_source = r"C:\Users\Pham Nhu Nguyen\Downloads\6880825981086.mp4"
        
        # Khởi tạo database
        BottleDB.init_db()

        # Stats TabView 
        self.stats_tabview = StatsTabView(self)
        self.stats_tabview.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)
        
        # Video Frame
        self.video_frame = VidFrame(
            self, 
            video_source=video_source,
            stats_callback=self.stats_tabview.update_stats,
            db=BottleDB,
            uart_callback=self.stats_tabview.send_uart
        )
        self.video_frame.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)
    
    def set_appearance_mode(self):
        """Toggle appearance mode between Light and Dark"""
        current_mode = ctk.get_appearance_mode()
        new_mode = "Dark" if current_mode == "Light" else "Light"
        ctk.set_appearance_mode(new_mode)
    
if __name__ == "__main__":
    try:
        is_dark = darkdetect.isDark()
    except Exception:
        is_dark = False
    app = App(is_dark)
    app.mainloop()