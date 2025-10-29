import customtkinter as ctk
from settings import *
from PIL import Image, ImageTk
import os
import tkinter as tk

class LoginScreen(ctk.CTkToplevel):
    def __init__(self, parent, on_login_success):
        super().__init__(parent)
        
        self.on_login_success = on_login_success
        self.parent = parent
        self.password_visible = False
        
        # Tài khoản mật khẩu cố định
        self.VALID_EMAIL = "nguyen"
        self.VALID_PASSWORD = "123"
        
        # Image attributes
        self.image = None
        self.image_ratio = None
        self.canvas_width = 0
        self.canvas_height = 0
        self.image_width = 0
        self.image_height = 0
        
        # setup
        self.title("Đăng Nhập")
        self.geometry("900x600")
        try:
            self.iconbitmap("water-bottle.ico")
        except:
            pass
        self.resizable(False, False)
        
        # Make this window modal
        self.transient(parent)
        self.grab_set()
        
        # Center the window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.winfo_screenheight() // 2) - (600 // 2)
        self.geometry(f'900x600+{x}+{y}')
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main container
        main_container = ctk.CTkFrame(self, fg_color=BACKGROUND_COLOR)
        main_container.pack(fill='both', expand=True)
        
        # Configure grid
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)  # Left side for image
        main_container.columnconfigure(1, weight=1)  # Right side for form
        
        # LEFT SIDE - Image
        left_frame = ctk.CTkFrame(main_container, fg_color=BACKGROUND_COLOR)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        
        try:
            # Load and display image
            image_path = os.path.join("images", "login_image.png")
            if os.path.exists(image_path):
                # Load original image
                self.image = Image.open(image_path)
                self.image_ratio = self.image.width / self.image.height
                
                # Create canvas for image
                self.image_canvas = tk.Canvas(
                    left_frame, 
                    bg=BACKGROUND_COLOR,
                    highlightthickness=0
                )
                self.image_canvas.pack(fill='both', expand=True)
                
                # Bind resize event
                self.image_canvas.bind('<Configure>', self.resize_image)
            else:
                # Placeholder if image not found
                placeholder_label = ctk.CTkLabel(
                    left_frame,
                    text="🖼️\nLogin Image",
                    font=ctk.CTkFont(size=48),
                    text_color=GREY
                )
                placeholder_label.pack(fill='both', expand=True)
        except Exception as e:
            # Error handling
            error_label = ctk.CTkLabel(
                left_frame,
                text=f"⚠️\nImage Error",
                font=ctk.CTkFont(size=32),
                text_color=GREY
            )
            error_label.pack(fill='both', expand=True)
        
        # RIGHT SIDE - Login Form
        right_frame = ctk.CTkFrame(main_container, fg_color=BACKGROUND_COLOR)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=30, pady=30)
        
        # Center the form vertically
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=1)
        
        # Top spacer
        ctk.CTkLabel(right_frame, text="").grid(row=0, column=0)
        
        # Form container
        form_frame = ctk.CTkFrame(right_frame, fg_color="transparent")
        form_frame.grid(row=1, column=0, sticky='ew', padx=20)
        
        # Title
        title_label = ctk.CTkLabel(
            form_frame, 
            text="ĐĂNG NHẬP", 
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=WHITE
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ctk.CTkLabel(
            form_frame,
            text="Hệ thống nhúng phân loại sản phẩm nước đóng chai\nbằng học máy và xử lý ảnh",
            font=ctk.CTkFont(size=16),
            text_color=GREY,
            justify='center'
        )
        subtitle_label.pack(pady=(0, 40))
        
        # Email
        email_label = ctk.CTkLabel(
            form_frame,
            text="Email:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=WHITE,
            anchor='w'
        )
        email_label.pack(pady=(10, 5), fill='x')
        
        self.email_entry = ctk.CTkEntry(
            form_frame,
            width=350,
            height=45,
            placeholder_text="Nhập email của bạn",
            font=ctk.CTkFont(size=14),
            border_width=2,
            corner_radius=8
        )
        self.email_entry.pack(pady=(0, 20), fill='x')
        self.email_entry.bind("<Return>", lambda e: self.password_entry.focus())
        
        # Password
        password_label = ctk.CTkLabel(
            form_frame,
            text="Mật khẩu:",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=WHITE,
            anchor='w'
        )
        password_label.pack(pady=(10, 5), fill='x')
        
        # Password frame (để chứa entry và nút show/hide)
        password_frame = ctk.CTkFrame(form_frame, fg_color="transparent")
        password_frame.pack(pady=(0, 10), fill='x')
        
        self.password_entry = ctk.CTkEntry(
            password_frame,
            width=315,
            height=45,
            placeholder_text="Nhập mật khẩu",
            show="●",
            font=ctk.CTkFont(size=14),
            border_width=2,
            corner_radius=8
        )
        self.password_entry.pack(side='left', padx=(0, 10), fill = 'x')
        self.password_entry.bind("<Return>", lambda e: self.login())
        
        # Show/Hide password button
        self.show_password_btn = ctk.CTkButton(
            password_frame,
            text="👁",
            width=45,
            height=45,
            font=ctk.CTkFont(size=20),
            fg_color=DARK_GREY,
            hover_color=GREY,
            corner_radius=8,
            command=self.toggle_password_visibility
        )
        self.show_password_btn.pack(side='right')
        
        # Error message label
        self.error_label = ctk.CTkLabel(
            form_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=CLOSE_RED
        )
        self.error_label.pack(pady=(10, 20))
        
        # Login button
        login_button = ctk.CTkButton(
            form_frame,
            text="Đăng Nhập",
            width=350,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color=BLUE,
            hover_color=DARK_GREY,
            corner_radius=10,
            command=self.login
        )
        login_button.pack(pady=(10, 20))
        
        # Bottom spacer
        ctk.CTkLabel(right_frame, text="").grid(row=2, column=0)
        
        # Focus on email entry
        self.email_entry.focus()
    
    def toggle_password_visibility(self):
        """Toggle password visibility"""
        if self.password_visible:
            self.password_entry.configure(show="●")
            self.show_password_btn.configure(text="👁")
            self.password_visible = False
        else:
            self.password_entry.configure(show="")
            self.show_password_btn.configure(text="🙈")
            self.password_visible = True
        
    def login(self):
        email = self.email_entry.get().strip()
        password = self.password_entry.get()
        
        if not email or not password:
            self.show_error("Vui lòng nhập đầy đủ thông tin!")
            return
        
        if email == self.VALID_EMAIL and password == self.VALID_PASSWORD:
            self.error_label.configure(text="Đăng nhập thành công!", text_color="#00ff00")
            self.after(500, self.on_success)
        else:
            self.show_error("Email hoặc mật khẩu không đúng!")
            self.password_entry.delete(0, 'end')
            self.password_entry.focus()
    
    def on_success(self):
        self.grab_release()
        self.destroy()
        self.on_login_success()
    
    def show_error(self, message):
        self.error_label.configure(text=message, text_color=CLOSE_RED)
    
    def resize_image(self, event):
        """Resize image to fill canvas without distortion"""
        if not self.image:
            return
        
        # Current canvas ratio
        canvas_ratio = event.width / event.height

        # Update canvas attributes
        self.canvas_width = event.width
        self.canvas_height = event.height

        # Resize based on ratio to fill the space
        if canvas_ratio > self.image_ratio:
            self.image_width = int(event.width)
            self.image_height = int(self.image_width / self.image_ratio)
        else:
            self.image_height = int(event.height)
            self.image_width = int(self.image_height * self.image_ratio)
        
        self.place_image()
    
    def place_image(self):
        """Place the resized image on canvas"""
        if not self.image:
            return
        
        # Resize the image
        resized_image = self.image.resize((self.image_width, self.image_height), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and place image centered
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            self.canvas_width / 2,
            self.canvas_height / 2,
            image=self.image_tk,
            anchor='center'
        )
