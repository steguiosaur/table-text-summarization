# vim: set textwidth=0:
from tkinter import Tk, PhotoImage
from PIL import Image
from customtkinter import (
    CTkImage,
    CTkLabel,
    CTkButton,
    CTkFrame,
    CTkOptionMenu,
    CTkTextbox,
    CTkToplevel,
    StringVar,
    set_appearance_mode,
    set_default_color_theme,
    set_widget_scaling,
)
from utils.pathing import Pathing
from preprocess.preprocess import Preprocess
from models.modelloader import ModelLoader
from rouge_score import rouge_scorer
from utils import npncns, settings


class Main(Tk):
    model_loader = None

    def __init__(self):
        super().__init__()
        self.configure(bg="#222222")
        self.grid_rowconfigure((8, 9), weight=1)
        self.grid_columnconfigure((0, 2), weight=1)

        # create a frame to contain the label and textbox together
        self.frame_input = CTkFrame(self, fg_color="#222222")
        self.frame_input.grid(row=0, column=0, rowspan=9, columnspan=3, sticky="nsew")
        self.label_input = CTkLabel(self.frame_input, text="INPUT TEXT AND TABLE", fg_color="#222222")
        self.label_input.pack(side="top", anchor="nw", padx=20, pady=(10, 0))
        self.textbox_input = CTkTextbox(
            self.frame_input, fg_color="#2e2e2e", font=("Consolas", 12), wrap="word"
        )
        self.textbox_input.pack(fill="both", expand=True, padx=20, pady=(0, 0))

        # show output and error messages
        self.frame_output = CTkFrame(self, fg_color="#222222")
        self.frame_output.grid(row=9, column=0, columnspan=3, sticky="nsew")
        self.label_output = CTkLabel(self.frame_output, text="GENERATED SUMMARY", fg_color="#222222")
        self.label_output.pack(side="top", anchor="nw", padx=20, pady=(10, 0))
        self.textbox_output = CTkTextbox(self.frame_output, height=100, fg_color="#2e2e2e", font=("Consolas", 12), wrap="word")
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
        self.option_model = CTkOptionMenu(self, values=["bart-lf-summ",], command=self.load_model)
        self.option_model.grid(row=2, column=4, padx=(0, 20), pady=0, sticky="")

        # change scaling of UI
        self.label_lang = CTkLabel(self, text="Change Scaling", fg_color="#222222")
        self.label_lang.grid(row=3, column=4, padx=(0, 20), pady=(10, 0), sticky="")
        self.currentScaling = StringVar(value=settings.scale_read())
        self.option_lang = CTkOptionMenu(self, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event, variable=self.currentScaling)
        self.option_lang.grid(row=4, column=4, padx=(0, 20), pady=(0, 20), sticky="")

        # evaluate button
        self.button_evaluation = CTkButton(self, text="Evaluate",
            command=lambda: self.show_eval_window(self.textbox_output.get("1.0", "end-1c"))
        )
        self.button_evaluation.grid(row=5, column=4, padx=(0, 20), pady=(20, 10), sticky="")

        # clear button
        self.button_clear = CTkButton(self, text="Clear Output", fg_color="#990000", hover_color="#770000", 
            command=self.delete_output
        )
        self.button_clear.grid(row=6, column=4, padx=(0, 20), pady=(20, 10), sticky="")

        # reset button
        self.button_reset = CTkButton(self, text="Clear All", fg_color="#990000", hover_color="#770000",
            command=self.delete_input_output
        )
        self.button_reset.grid(row=7, column=4, padx=(0, 20), pady=20, sticky="")

        # go button
        self.button_start = CTkButton(master=self, text="Generate",
            command=self.generate_summary
        )
        self.button_start.grid(row=9, column=4, padx=(0, 20), pady=(40, 20), sticky="nsew")

        self.load_model()

    def change_scaling_event(self, new_scaling: str):
        # change scaling of application
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        set_widget_scaling(new_scaling_float)
        settings.scale_save(new_scaling_float)

    def show_eval_window(self, generated_summary):
        # Create a frame that overlays on top of frame_input and frame_output
        self.eval_frame = CTkFrame(self, fg_color="#222222", corner_radius=0)
        self.eval_frame.grid(row=0, column=0, rowspan=10, columnspan=3, padx=(0, 10), sticky="nsew")
        self.eval_frame.rowconfigure([2], weight=1)
        self.eval_frame.columnconfigure([0, 3], weight=1)

        # Textbox to display target summary
        label_target_text = CTkLabel(self.eval_frame, text="TARGET SUMMARY", fg_color="#222222")
        label_target_text.grid(row=0, column=0, padx=20, pady=(10, 0), sticky="nw")
        self.textbox_target_text = CTkTextbox(self.eval_frame, fg_color="#2e2e2e", font=("Consolas", 12), wrap="word")
        self.textbox_target_text.grid(row=1, column=0, rowspan=2, columnspan=2, padx=(20, 10), pady=(0, 10), sticky="nsew")

        # Textbox to display generated summary
        label_summary_eval = CTkLabel(self.eval_frame, text="GENERATED SUMMARY:", fg_color="#222222")
        label_summary_eval.grid(row=0, column=2, padx=20, pady=(10, 0), sticky="nw")
        self.textbox_summary_eval = CTkTextbox(self.eval_frame, fg_color="#2e2e2e", font=("Consolas", 12), wrap="word")
        self.textbox_summary_eval.insert("1.0", generated_summary)
        self.textbox_summary_eval.grid(row=1, column=2, rowspan=2, columnspan=2, padx=(10, 20), pady=(0, 10), sticky="nsew")

        # Frame for evaluations
        eval_results_frame = CTkFrame(self.eval_frame, fg_color="#2e2e2e", corner_radius=0)
        eval_results_frame.grid(row=3, column=0, columnspan=4, padx=20, pady=(10, 20), sticky="sew")

        # Store eval_results_frame for reuse
        self.eval_results_frame = eval_results_frame

        # default ROUGE scores (replaces these with actual computed values)
        rouge_scores = {
            "ROUGE-1": {"Precision": 0.00, "Recall": 0.00, "Fmeasure": 0.00},
            "ROUGE-2": {"Precision": 0.00, "Recall": 0.00, "Fmeasure": 0.00},
            "ROUGE-L": {"Precision": 0.00, "Recall": 0.00, "Fmeasure": 0.00},
        }

        # Add evaluations to the frame
        eval_results_frame.columnconfigure((0, 1, 2, 3), weight=1)

        for col, (rouge_type, metrics) in enumerate(rouge_scores.items()):
            label_rouge_type = CTkLabel(eval_results_frame, text=rouge_type, font=("Consolas", 12, "bold"))
            label_rouge_type.grid(row=0, column=col, padx=10, pady=10, sticky="n")

            for row, (metric, score) in enumerate(metrics.items(), start=1):
                label_metric = CTkLabel(
                    eval_results_frame,
                    text=f"{metric}: {score:.2f}",
                    font=("Consolas", 12),
                )
                label_metric.grid(row=row, column=col, padx=10, pady=(0, 10), sticky="n")

        # Populate Numerical Measures
        numerical_measures = {"NP": 0.00, "NC": 0.00, "NS": 0.00}
        numerical_column = len(rouge_scores)
        label_numerical_type = CTkLabel(
            eval_results_frame, text="Numerical Measures", font=("Consolas", 12, "bold")
        )
        label_numerical_type.grid(row=0, column=numerical_column, padx=10, pady=10, sticky="n")

        for row, (measure, value) in enumerate(numerical_measures.items(), start=1):
            label_measure = CTkLabel(
                eval_results_frame,
                text=f"{measure}: {value:.2f}",
                font=("Consolas", 12),
            )
            label_measure.grid(row=row, column=numerical_column, padx=10, pady=(0, 10), sticky="n")

        # Change buttons on the right side
        self.update_right_side_buttons()

    def evaluate_summary(self):
        # Extract summaries from stored textboxes
        input_text = self.textbox_input.get("1.0", "end-1c").strip()
        generated_summary = self.textbox_summary_eval.get("1.0", "end-1c").strip()
        target_summary = self.textbox_target_text.get("1.0", "end-1c").strip()

        # Check if target summary is empty
        if not target_summary:
            self.show_popup("Error", "Target summary cannot be empty.")
            return
        if not generated_summary:
            self.show_popup("Error", "Generated summary cannot be empty.")
            return

        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(target_summary, generated_summary)

        # Extract numbers from input, generated, and target summaries
        D_n = npncns.extract_numbers(input_text)
        H_n = npncns.extract_numbers(generated_summary)
        S_n = npncns.extract_numbers(target_summary)

        # Calculate numerical measures
        np = npncns.calculate_np(H_n, S_n)
        nc = npncns.calculate_nc(D_n, H_n, S_n)
        ns = npncns.calculate_ns(np, nc)

        # Update the eval_results_frame
        for col, rouge_type in enumerate(['rouge1', 'rouge2', 'rougeL']):
            score = scores[rouge_type]  # This is a named tuple with precision, recall, and fmeasure
            for row, metric in enumerate(['precision', 'recall', 'fmeasure'], start=1):
                score_value = getattr(score, metric) * 100  # Access the attribute by name
                label_metric = self.eval_results_frame.grid_slaves(row=row, column=col)[0]
                label_metric.configure(text=f"{metric.capitalize()}: {score_value:.2f}")

        # Update the eval_results_frame with numerical measures
        numerical_measures = {"NP": np, "NC": nc, "NS": ns}
        numerical_column = len(scores)  # This is the fourth column (index 3)

        for row, (measure, value) in enumerate(numerical_measures.items(), start=1):
            label_measure = self.eval_results_frame.grid_slaves(row=row, column=numerical_column)[0]
            label_measure.configure(text=f"{measure}: {value:.2f}")

    def show_popup(self, title, message):
        popup = CTkToplevel(self)
        popup.title(title)
        label_message = CTkLabel(popup, text=message, font=("Consolas", 12))
        label_message.pack(pady=10, padx=20)
        button_ok = CTkButton(popup, text="OK", command=popup.destroy)
        button_ok.pack(pady=(0, 10))

    def hide_eval_frame(self):
        self.eval_frame.destroy()
        self.restore_right_side_buttons()

    def update_right_side_buttons(self):
        self.button_evaluation.configure(text="Summarize", command=self.hide_eval_frame)
        self.button_clear.configure(command=lambda: self.textbox_target_text.delete("0.0", "end"))
        self.button_reset.configure(command=lambda: self.textbox_target_text.delete("0.0", "end"))
        self.button_start.configure(text="Evaluate", command=self.evaluate_summary)

    def restore_right_side_buttons(self):
        self.button_evaluation.configure(text="Evaluate", command=lambda: self.show_eval_window(self.textbox_output.get("1.0", "end-1c")))
        self.button_clear.configure(command=self.delete_output)
        self.button_reset.configure(command=self.delete_input_output)
        self.button_start.configure(text="Generate", command=self.generate_summary)

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

    def delete_output(self):
        self.textbox_output.delete("0.0", "end")


# initialize settings and themes
settings.initialize_config()
set_appearance_mode(settings.appearance_read())
set_default_color_theme(settings.theme_read())
set_widget_scaling(settings.int_scale_read())

app = Main()
app.withdraw()
app.title("T&TSumm")
app.resizable(True, True)
width = 1024
height = 576
app.deiconify()
x = (app.winfo_screenwidth() // 2) - width // 2
y = (app.winfo_screenheight() // 2) - height // 2
app.geometry("%dx%d+%d+%d" % (width, height, x, y))
app.minsize(1024, 576)
app.iconphoto(True, PhotoImage(file=Pathing.asset_path("tbldsc_white_logo.png")))
app.mainloop()
