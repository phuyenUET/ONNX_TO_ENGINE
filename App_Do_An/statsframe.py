import customtkinter as ctk
from settings import *
import serial
import serial.tools.list_ports
from tkinter import messagebox

class StatsTabView(ctk.CTkTabview):
    """TabView chứa Stats và UART Settings"""
    def __init__(self, parent):
        super().__init__(master=parent)
        
        # Tạo 2 tabs
        self.add("STATS")
        self.add("SETTINGS")
        
        # Stats Frame trong tab 1
        self.stats_frame = StatsFrame(self.tab("STATS"))

        # UART Settings trong tab 2
        self.uart_frame = UARTFrame(self.tab("SETTINGS"))

    def update_stats(self, stats):
        """Wrapper để update stats"""
        self.stats_frame.update_stats(stats)
    
    def send_uart(self, error_code: str):
        """Gửi error code qua UART (được gọi từ VidFrame)"""
        self.uart_frame.send_error_code(error_code)  


class StatsFrame(ctk.CTkFrame):
    """Frame hiển thị thống kê"""
    def __init__(self, parent):
        super().__init__(parent, fg_color=DARK_GREY, corner_radius=10)
        self.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title = ctk.CTkLabel(
            self,
            text="📊 THỐNG KÊ",
            font=TITLE_FONT,
            text_color=WHITE
        )
        title.pack(pady=(20, 10))
        
        # Separator
        separator = ctk.CTkFrame(self, height=2, fg_color=BLUE)
        separator.pack(fill='x', padx=20, pady=10)
        
        # Stats container với grid layout
        stats_container = ctk.CTkFrame(self, fg_color=DARK_GREY)
        stats_container.pack(fill='x', padx=20, pady=10)
        
        # Configure grid columns
        stats_container.columnconfigure(0, weight=1)  # Label column
        stats_container.columnconfigure(1, weight=0)  # Number column (fixed width)
        
        # Row 0: Tổng số
        total_icon = ctk.CTkLabel(
            stats_container,
            text="📦  TỔNG SỐ CHAI:",
            font=MAIN_FONT,
            text_color=WHITE,
            anchor='w'
        )
        total_icon.grid(row=0, column=0, sticky='w', pady=5)
        
        self.total_value = ctk.CTkLabel(
            stats_container,
            text="0",
            font=BOLD_FONT,
            text_color=WHITE,
            width=60,
            anchor='e'
        )
        self.total_value.grid(row=0, column=1, sticky='e', pady=5, padx=(10, 0))
        
        # Row 1: Chai OK
        ok_icon = ctk.CTkLabel(
            stats_container,
            text="✅ CHAI OK (0):",
            font=MAIN_FONT,
            text_color=GREEN,
            anchor='w'
        )
        ok_icon.grid(row=1, column=0, sticky='w', pady=5)
        
        self.ok_value = ctk.CTkLabel(
            stats_container,
            text="0",
            font=BOLD_FONT,
            text_color=GREEN,
            width=60,
            anchor='e'
        )
        self.ok_value.grid(row=1, column=1, sticky='e', pady=5, padx=(10, 0))
        
        # Row 2: Lỗi mực nước
        water_icon = ctk.CTkLabel(
            stats_container,
            text="💧  LỖI MỰC NƯỚC (1):",
            font=MAIN_FONT,
            text_color=YELLOW,
            anchor='w'
        )
        water_icon.grid(row=2, column=0, sticky='w', pady=5)
        
        self.water_value = ctk.CTkLabel(
            stats_container,
            text="0",
            font=BOLD_FONT,
            text_color=YELLOW,
            width=60,
            anchor='e'
        )
        self.water_value.grid(row=2, column=1, sticky='e', pady=5, padx=(10, 0))
        
        # Row 3: Lỗi nhãn
        label_icon = ctk.CTkLabel(
            stats_container,
            text="🔖  LỖI NHÃN (2):",
            font=MAIN_FONT,
            text_color=RED,
            anchor='w'
        )
        label_icon.grid(row=3, column=0, sticky='w', pady=5)
        
        self.label_value = ctk.CTkLabel(
            stats_container,
            text="0",
            font=BOLD_FONT,
            text_color=RED,
            width=60,
            anchor='e'
        )
        self.label_value.grid(row=3, column=1, sticky='e', pady=5, padx=(10, 0))
        
        # Separator
        separator2 = ctk.CTkFrame(self, height=2, fg_color=BLUE)
        separator2.pack(fill='x', padx=20, pady=20)
        
        # Current bottles title
        current_title = ctk.CTkLabel(
            self,
            text="🔍 CHAI ĐANG KIỂM TRA",
            font=BOLD_FONT,
            text_color=WHITE
        )
        current_title.pack(pady=(0, 10))
        
        # Scrollable frame for current bottles
        self.bottles_frame = ctk.CTkScrollableFrame(
            self,
            fg_color=BACKGROUND_COLOR,
            height=250
        )
        self.bottles_frame.pack(fill='both', expand=True, padx=10, pady=(0, 20))
        
        self._update_scheduled = False  # Để throttle updates
    
    def update_stats(self, stats):
        """Cập nhật thống kê - throttled"""
        if self._update_scheduled:
            return
        
        self._update_scheduled = True
        self.after(50, lambda: self._do_update(stats))  # Update mỗi 50ms
    
    def _do_update(self, stats):
        """Thực hiện update"""
        self._update_scheduled = False
        
        # Update số liệu
        self.total_value.configure(text=str(stats['total']))
        self.ok_value.configure(text=str(stats['ok']))
        self.water_value.configure(text=str(stats['water_error']))
        self.label_value.configure(text=str(stats['label_error']))
        
        # Clear và update bottles
        for widget in self.bottles_frame.winfo_children():
            widget.destroy()
        
        if not stats['current_bottles']:
            no_bottle_label = ctk.CTkLabel(
                self.bottles_frame,
                text="Không có chai nào",
                font=MAIN_FONT,
                text_color=GREY
            )
            no_bottle_label.pack(pady=10)
        else:
            for bottle_id, water_ok, label_ok, error_type in stats['current_bottles']:
                # Tạo frame cho từng bottle với grid layout
                bottle_frame = ctk.CTkFrame(
                    self.bottles_frame,
                    fg_color=DARK_GREY,
                    corner_radius=5
                )
                bottle_frame.pack(fill='x', pady=5, padx=5)
                
                # Configure grid
                bottle_frame.columnconfigure(0, weight=0)  # ID
                bottle_frame.columnconfigure(1, weight=1)  # Nước
                bottle_frame.columnconfigure(2, weight=1)  # Nhãn
                
                # ID
                id_label = ctk.CTkLabel(
                    bottle_frame,
                    text=f"ID: {bottle_id}",
                    font=BOLD_FONT,
                    text_color=BLUE,
                    width=70,
                    anchor='w'
                )
                id_label.grid(row=0, column=0, padx=10, pady=8, sticky='w')
                
                # Water status
                water_icon = "✅" if water_ok else "❌"
                water_label = ctk.CTkLabel(
                    bottle_frame,
                    text=f"{water_icon} Nước",
                    font=MAIN_FONT,
                    text_color=GREEN if water_ok else RED,
                    anchor='center'
                )
                water_label.grid(row=0, column=1, padx=5, pady=8)
                
                # Label status
                label_icon = "✅" if label_ok else "❌"
                label_label = ctk.CTkLabel(
                    bottle_frame,
                    text=f"{label_icon} Nhãn",
                    font=MAIN_FONT,
                    text_color=GREEN if label_ok else RED,
                    anchor='center'
                )
                label_label.grid(row=0, column=2, padx=5, pady=8)


class UARTFrame(ctk.CTkFrame):
    """Frame cài đặt UART"""
    def __init__(self, parent):
        super().__init__(parent, fg_color=DARK_GREY, corner_radius=10)
        self.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.ser = None
        
        # Title
        title = ctk.CTkLabel(
            self,
            text="⚙️ CÀI ĐẶT UART",
            font=TITLE_FONT,
            text_color=WHITE
        )
        title.pack(pady=(20, 10))
        
        # Separator
        separator = ctk.CTkFrame(self, height=2, fg_color=BLUE)
        separator.pack(fill='x', padx=20, pady=10)
        
        # Settings container
        settings_frame = ctk.CTkFrame(self, fg_color=DARK_GREY)
        settings_frame.pack(fill='x', padx=20, pady=10)
        
        # COM Port
        port_label = ctk.CTkLabel(
            settings_frame,
            text="Chọn COM Port:",
            font=MAIN_FONT,
            text_color=WHITE
        )
        port_label.pack(pady=(10, 5))
        
        self.combo_port = ctk.CTkComboBox(
            settings_frame,
            values=self.scan_ports(),
            width=200,
            font=MAIN_FONT
        )
        self.combo_port.pack(pady=(0, 15))
        
        # Baudrate
        baud_label = ctk.CTkLabel(
            settings_frame,
            text="Chọn Baudrate:",
            font=MAIN_FONT,
            text_color=WHITE
        )
        baud_label.pack(pady=(10, 5))
        
        self.combo_baud = ctk.CTkComboBox(
            settings_frame,
            values=["9600", "115200", "57600"],
            width=200,
            font=MAIN_FONT
        )
        self.combo_baud.set("115200")
        self.combo_baud.pack(pady=(0, 15))
        
        # Buttons frame
        btn_frame = ctk.CTkFrame(settings_frame, fg_color=DARK_GREY)
        btn_frame.pack(pady=10)

        self.btn_connect = ctk.CTkButton(
            btn_frame, text="Kết nối", command=self.connect_serial,
            font=BOLD_FONT, width=120, fg_color=GREEN, hover_color="#27ae60"
        )
        self.btn_connect.grid(row=0, column=0, padx=5)

        self.btn_disconnect = ctk.CTkButton(
            btn_frame, text="Ngắt kết nối", command=self.disconnect_serial,
            font=BOLD_FONT, width=120, fg_color=RED, hover_color="#c0392b"
        )
        self.btn_disconnect.grid(row=0, column=1, padx=5)
        
        # Status label
        self.lbl_status = ctk.CTkLabel(
            settings_frame,
            text="Chưa kết nối",
            font=BOLD_FONT,
            text_color=RED
        )
        self.lbl_status.pack(pady=10)
        
        # Separator
        separator2 = ctk.CTkFrame(self, height=2, fg_color=BLUE)
        separator2.pack(fill='x', padx=20, pady=20)
    
    def scan_ports(self):
        """Quét các COM port có sẵn"""
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports] if ports else ["Không có COM"]
    
    # def refresh_ports(self):
    #     """Làm mới danh sách COM ports"""
    #     ports = self.scan_ports()
    #     self.combo_port.configure(values=ports)
    #     if ports and ports[0] != "Không có COM":
    #         self.combo_port.set(ports[0])
    
    def connect_serial(self):
        """Kết nối UART"""
        port = self.combo_port.get()
        baud = self.combo_baud.get()
        
        if not port or port == "Không có COM":
            messagebox.showwarning("Thiếu thông tin", "Chọn COM port hợp lệ")
            return
        
        if not baud:
            messagebox.showwarning("Thiếu thông tin", "Chọn Baudrate")
            return
        
        try:
            self.ser = serial.Serial(port, int(baud), timeout=1)
            self.lbl_status.configure(
                text=f"✅ Đã kết nối {port}",
                text_color=GREEN
            )
            messagebox.showinfo("Thành công", f"Đã kết nối {port} @ {baud} baud")
        except Exception as e:
            messagebox.showerror("Lỗi kết nối", str(e))
            self.lbl_status.configure(
                text="❌ Kết nối thất bại",
                text_color=RED
            )
    
    def disconnect_serial(self):
        """Ngắt kết nối UART"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.lbl_status.configure(
                text="⚠️ Đã ngắt kết nối",
                text_color=YELLOW
            )
            messagebox.showinfo("Thông báo", "Đã ngắt kết nối UART")
        else:
            messagebox.showinfo("Thông báo", "Chưa kết nối thiết bị nào")
    
    def send_error_code(self, code: str):
        """Gửi mã lỗi qua UART (chỉ dùng tự động, không popup)."""
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(code.encode())
                print(f"[UART] sent: {code}")
            except Exception as e:
                print(f"[UART] send failed: {e}")
        else:
            print("[UART] not connected")
    
    def __del__(self):
        """Cleanup khi destroy"""
        if self.ser and self.ser.is_open:
            self.ser.close()