import customtkinter as ctk
import tkinter as tk
from settings import *

class AboutWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Setup window
        self.title("Thông tin đồ án")
        self.geometry("600x400")
        try:
            self.iconbitmap("water-bottle.ico")
        except:
            pass
        self.resizable(False, False)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (600 // 2)
        y = (self.winfo_screenheight() // 2) - (400 // 2)
        self.geometry(f'600x400+{x}+{y}')
        
        # Create content
        self.create_widgets()
    
    def create_widgets(self):
        # Main frame
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)
        
        # Title
        title_label = ctk.CTkLabel(
            main_frame,
            text="Thông Tin Đồ Án",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=BLUE
        )
        title_label.pack(pady=(0, 20))
        
        # Subtitle
        subtitle_label = ctk.CTkLabel(
            main_frame,
            text="Thử nghiệm hệ thống nhúng phân loại sản phẩm \nnước đóng chai bằng học máy và xử lý ảnh",
            font=ctk.CTkFont(size=14, weight="bold"),
            justify='center'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Description frame with scrollbar
        description_frame = ctk.CTkFrame(main_frame, fg_color=DARK_GREY, corner_radius=10)
        description_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Description text
        description_text = ctk.CTkTextbox(
            description_frame,
            font=ctk.CTkFont(size=13),
            wrap='word',
            fg_color=DARK_GREY,
            corner_radius=10,
            border_width=0
        )
        description_text.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Insert content
        content = """Đồ án tập trung xây dựng một hệ thống giám sát và phân loại lỗi sản phẩm trên băng chuyền dựa trên thị giác máy tính và điều khiển nhúng. Hệ thống có khả năng nhận diện, tracking và phát hiện lỗi chai nước theo thời gian thực, đồng thời điều khiển cơ cấu phân loại để loại bỏ sản phẩm không đạt. Dữ liệu kiểm tra được lưu trữ vào cơ sở dữ liệu, hiển thị qua giao diện trực quan.

        ────────────────────────────────────

        👤 Thông tin sinh viên:
        • Tên: Phạm Như Nguyên
        • MSV: 21020496
        • Lớp: K66M-AT"""
        
        description_text.insert("1.0", content)
        description_text.configure(state="disabled")  # Read-only

class Menu(tk.Menu):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.parent = parent
        
        # Mode
        mode_menu = tk.Menu(self, tearoff=0)
        help_menu = tk.Menu(self, tearoff=0)
        help_menu.add_command(label="About This Project", command=self.show_about)
        mode_menu.add_checkbutton(label="Change Appearance Mode", command=parent.set_appearance_mode)
        self.add_cascade(label="Mode", menu=mode_menu)
        self.add_cascade(label="About", menu=help_menu)
    
    def show_about(self):
        """Show About window"""
        AboutWindow(self.parent)
    
    