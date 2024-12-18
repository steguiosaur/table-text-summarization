from tkinter import Tk, PhotoImage
from PIL import Image
from customtkinter import (
    CTkImage,
    CTkLabel,
    CTkButton,
    CTkFrame,
    CTkOptionMenu,
    CTkTextbox,
    set_appearance_mode,
    set_default_color_theme,
    set_widget_scaling,
)
from utils.pathing import Pathing
from preprocess.preprocess import Preprocess
from models.modelloader import ModelLoader


class Main(Tk):
    model_loader = None

    def __init__(self):
        super().__init__()

        self.model = None
        self.tokenizer = None

        self.configure(bg="#222222")

        self.grid_rowconfigure((7, 8), weight=1)
        self.grid_columnconfigure((0, 2), weight=1)

        # create a frame to contain the label and textbox together
        self.frame_input = CTkFrame(self, fg_color="#222222")
        self.frame_input.grid(row=0, column=0, rowspan=8, columnspan=3, sticky="nsew")
        self.label_input = CTkLabel(
            self.frame_input, text="INPUT TEXT AND TABLE", fg_color="#222222"
        )
        self.label_input.pack(side="top", anchor="nw", padx=20, pady=(10, 0))
        self.textbox_input = CTkTextbox(
            self.frame_input, fg_color="#2e2e2e", font=("Consolas", 12), wrap="word"
        )
        self.textbox_input.pack(fill="both", expand=True, padx=20, pady=(0, 0))

        # show output and error messages
        self.frame_output = CTkFrame(self, fg_color="#222222")
        self.frame_output.grid(row=8, column=0, columnspan=3, sticky="nsew")
        self.label_output = CTkLabel(
            self.frame_output, text="GENERATED SUMMARY", fg_color="#222222"
        )
        self.label_output.pack(side="top", anchor="nw", padx=20, pady=(10, 0))
        self.textbox_output = CTkTextbox(
            self.frame_output,
            height=100,
            fg_color="#2e2e2e",
            font=("Consolas", 12),
            wrap="word",
        )
        self.textbox_output.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # logo image
        self.image_logo = CTkImage(
            light_image=Image.open(Pathing.asset_path("t&t-Summ-black.png")),
            dark_image=Image.open(Pathing.asset_path("t&t-Summ-white.png")),
            size=(138, 38),
        )
        self.label_logo = CTkLabel(self, text="", image=self.image_logo)
        self.label_logo.grid(row=0, column=4, padx=(0, 20), pady=20, sticky="n")

        # model selector
        self.label_model = CTkLabel(self, text="Current Model", fg_color="#222222")
        self.label_model.grid(row=1, column=4, padx=(0, 20), pady=(20, 0), sticky="")
        self.option_model = CTkOptionMenu(
            self,
            values=[
                "bart-lf-summ",
            ],
            command=self.load_model,
        )
        self.option_model.grid(row=2, column=4, padx=(0, 20), pady=0, sticky="")

        # markdup language editor table selector
        self.label_lang = CTkLabel(self, text="Markup Language", fg_color="#222222")
        self.label_lang.grid(row=3, column=4, padx=(0, 20), pady=(10, 0), sticky="")
        self.option_lang = CTkOptionMenu(self, values=["Markdown"])
        self.option_lang.grid(row=4, column=4, padx=(0, 20), pady=(0, 20), sticky="")

        # import table button
        self.button_clear = CTkButton(
            self, text="Clear Output", command=self.delete_output
        )
        self.button_clear.grid(row=5, column=4, padx=(0, 20), pady=(20, 10), sticky="")

        # reset button
        self.button_reset = CTkButton(
            self, text="Clear All", command=self.delete_input_output
        )
        self.button_reset.grid(row=6, column=4, padx=(0, 20), pady=20, sticky="")

        # go button
        self.button_start = CTkButton(
            master=self, text="Generate", command=self.generate_summary
        )
        self.button_start.grid(
            row=8, column=4, padx=(0, 20), pady=(40, 20), sticky="nsew"
        )

        self.load_model()

    def load_model(self, selected_model=None):
        selected_model = selected_model or self.option_model.get()

        if selected_model == "bart-lf-summ":
            model_path = Pathing.model_path("bart-large_ep1.pt")
            self.model_loader = ModelLoader("facebook/bart-large", model_path)
            self.model, self.tokenizer = self.model_loader.tntsumm()
        else:
            print(f"Model '{selected_model}' is not configured.")

    def delete_input_output(self):
        self.textbox_input.delete("0.0", "end")
        self.textbox_output.delete("0.0", "end")

    def generate_summary(self):
        text_in = self.textbox_input.get("0.0", "end")
        preprocess_dict = Preprocess.parse_markdown(text_in)
        linearized_tbl = Preprocess.linearize_table_data(preprocess_dict)

        if self.model_loader:
            try:
                generate_output = self.model_loader.generate_output(
                    linearized_tbl["src_text"]
                )
                self.textbox_output.delete("1.0", "end")
                self.textbox_output.insert("1.0", generate_output)
            except Exception as e:
                self.textbox_output.delete("1.0", "end")
                self.textbox_output.insert("1.0", f"Error generating summary: {e}")
        else:
            self.textbox_output.delete("1.0", "end")
            self.textbox_output.insert(
                "1.0", "Model not loaded. Please select a model."
            )

    def show_splash_screen(self):
        self.splash_frame = CTkFrame(master=self, fg_color="#222222")
        self.splash_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.splash_image = CTkImage(
            light_image=Image.open(Pathing.asset_path("tbldsc_black.png")),
            dark_image=Image.open(Pathing.asset_path("tbldsc_white.png")),
            size=(276, 76),
        )
        self.splash_label = CTkLabel(
            self.splash_frame, text="", image=self.splash_image
        )
        self.splash_label.pack(expand=True)

        self.splash_frame.bind("<Button-1>", self.hide_splash_screen)
        self.splash_frame.focus_set()

    def hide_splash_screen(self, event=None):
        self.splash_frame.destroy()

    def delete_output(self):
        self.textbox_output.delete("0.0", "end")


set_appearance_mode("Dark")
set_default_color_theme("blue")
set_widget_scaling(1)

app = Main()
app.withdraw()
app.title("tbldsc")
app.resizable(True, True)
width = 1024
height = 576
app.deiconify()
x = (app.winfo_screenwidth() // 2) - width // 2
y = (app.winfo_screenheight() // 2) - height // 2
app.geometry("%dx%d+%d+%d" % (width, height, x, y))
app.minsize(1024, 576)
app.iconphoto(True, PhotoImage(file=Pathing.asset_path("tbldsc_white_logo.png")))
# app.show_splash_screen()
app.mainloop()
