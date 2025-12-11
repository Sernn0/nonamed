"""
Tkinter 기반 로컬 UI 예제

흐름:
1) 메인 화면: 간편/정밀 모드 선택 + 양식 다운로드 버튼
2) PDF 드롭/선택 화면: PDF 한 개 선택하면 바로 처리 시작
3) 로딩 화면: 간단한 로그/프로그레스 표시(실제 처리 루프에서 갱신)
4) 결과 화면: SVG/TTF 다운로드 버튼

주의:
- PDF→이미지, 모델 추론, SVG→TTF 변환은 실제 구현이 필요하며, 본 예제에서는 TODO/더미 처리.
- 필수 의존성은 시작 시 CLI 로그로 검사한다.
- potrace/fontmake(fontTools) 경로가 PATH에 있어야 한다. 미검출 시 경고만 출력.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import os
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 대용량 PDF 대비 Pillow 경고/차단 해제
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import PhotoImage

# 프로젝트 루트 기준 기본 경로
ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "ui" / "assets"
CHARSET_SIMPLE = ROOT / "charset_50.txt"
CHARSET_DETAILED = ROOT / "charset_220.txt"
OUTPUT_DIR = ROOT / "outputs"

# 프로젝트 루트를 sys.path에 추가 (src 모듈 import를 위해)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 파이프라인 모듈 (v2 - 폰트 렌더링 학습 모델)
PIPELINE_AVAILABLE = False
try:
    from src.inference.pipeline_v2 import generate_all_glyphs_v2
    from src.inference.pipeline import create_ttf_font  # TTF 생성은 기존 것 사용
    PIPELINE_AVAILABLE = True
    print("[UI] Pipeline v2 loaded successfully!")
except ImportError as e:
    print(f"[UI] Pipeline v2 import failed: {e}")

# Drag & Drop (필수 의존성)
from tkinterdnd2 import DND_FILES, TkinterDnD
HAS_DND = True
# 색상 팔레트 (디자인 시안과 유사하게 세팅)
COLOR_BG = "#ffffff"
COLOR_PANEL = "#e5e5e5"
COLOR_TEXT = "#111111"
COLOR_SUB = "#6a6a6a"
COLOR_BTN_GRAY = "#cfcfcf"
COLOR_BTN_BLACK = "#000000"


# ----------------------- 의존성 체크 ----------------------- #
def check_dependency(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False


def locate_poppler() -> Optional[str]:
    # 1) 환경변수
    env_path = os.environ.get("POPPLER_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    # 2) macOS brew 기본 경로들
    for guess in ("/opt/homebrew/opt/poppler/bin", "/usr/local/opt/poppler/bin"):
        if Path(guess).exists():
            return guess
    # 3) 시스템 PATH 내 pdftoppm
    which_bin = shutil.which("pdftoppm")
    if which_bin:
        return str(Path(which_bin).parent)
    return None


def run_preflight_checks() -> None:
    print("[INFO] Running preflight checks...")
    # Python 패키지
    py_deps = [
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("pdf2image", "pdf2image"),
        ("fontTools", "fontTools"),
        ("fontmake", "fontmake"),
        ("tkinterdnd2", "tkinterdnd2"),
    ]
    for name, mod in py_deps:
        try:
            __import__(mod)
            print(f"[OK] {name} installed")
        except ImportError:
            print(f"[WARN] {name} NOT found")
    # CLI 도구
    for tool in ["potrace", "fontmake"]:
        ok = check_dependency([tool, "--version"])
        print(f"[{'OK' if ok else 'WARN'}] {tool} detected" if ok else f"[WARN] {tool} not detected (optional)")
    poppler_path = locate_poppler()
    if poppler_path:
        print(f"[OK] poppler detected at {poppler_path}")
    else:
        print("[WARN] poppler not detected. PDF 변환을 위해 설치/경로 설정이 필요합니다.")
    print("[INFO] Preflight checks done.")


# ----------------------- 데이터 구조 ----------------------- #
@dataclass
class AppState:
    mode: str = "simple"  # "simple" or "detailed"
    pdf_path: Optional[Path] = None
    output_svg: Optional[Path] = None
    output_ttf: Optional[Path] = None

    @property
    def charset_file(self) -> Path:
        return CHARSET_SIMPLE if self.mode == "simple" else CHARSET_DETAILED


# ----------------------- 화면 베이스 ----------------------- #
class BaseFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, style="Main.TFrame")
        self.controller = controller


# ----------------------- 메인 화면 ----------------------- #
class MainPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.build_ui()

    def build_ui(self):
        # 상단 바
        top = ttk.Frame(self, style="Main.TFrame")
        top.pack(fill="x", pady=(8, 6), padx=10)
        ttk.Label(top, text="Font By Me", style="Title.TLabel").pack(side="left")
        ttk.Label(top, text="운여명   황경준   김수인", style="Names.TLabel").pack(side="right")

        # 좌:우 비율 4:6, 가운데에 여백 확보
        body = ttk.Frame(self, style="Main.TFrame")
        body.pack(fill="both", expand=True, padx=(12, 0), pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=3)
        body.rowconfigure(0, weight=1)

        # 왼쪽 영역: 모드 선택
        left_holder = ttk.Frame(body, style="Main.TFrame")
        left_holder.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        # 위/아래에 가상 여백을 두어 중앙 정렬 효과
        left_holder.rowconfigure(0, weight=1)
        left_holder.rowconfigure(2, weight=1)
        left_holder.columnconfigure(0, weight=1)

        mode_frame = ttk.Frame(left_holder, style="Main.TFrame")
        mode_frame.grid(row=1, column=0, sticky="n")
        mode_frame.columnconfigure(0, weight=0)
        mode_frame.columnconfigure(1, weight=0)
        btn_width = 14

        # 간편 모드
        ttk.Label(mode_frame, text="→ 간편 모드", style="BoldSmall.TLabel").grid(row=0, column=0, sticky="w", pady=(4, 6), padx=(0, 0), columnspan=2)
        ttk.Button(
            mode_frame,
            text="간편 모드 양식 다운로드",
            style="GraySmall.TButton",
            width=btn_width,
            command=lambda: self.download_form("simple"),
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(2, 10))
        ttk.Button(
            mode_frame,
            text="간편 모드로 시작",
            style="GraySmall.TButton",
            width=btn_width,
            command=lambda: self.start_mode("simple"),
        ).grid(row=1, column=1, sticky="w", padx=(0, 0), pady=(2, 10))

        # 정밀 모드
        ttk.Label(mode_frame, text="→ 정밀 모드", style="BoldSmall.TLabel").grid(row=2, column=0, sticky="w", pady=(6, 6), padx=(0, 0), columnspan=2)
        ttk.Button(
            mode_frame,
            text="정밀 모드 양식 다운로드",
            style="GraySmall.TButton",
            width=btn_width,
            command=lambda: self.download_form("detailed"),
        ).grid(row=3, column=0, sticky="w", padx=(0, 10), pady=(2, 16))
        ttk.Button(
            mode_frame,
            text="정밀 모드로 시작",
            style="GraySmall.TButton",
            width=btn_width,
            command=lambda: self.start_mode("detailed"),
        ).grid(row=3, column=1, sticky="w", padx=(0, 0), pady=(2, 16))

        ttk.Label(
            mode_frame,
            text="손글씨 PDF를 받아 폰트를 생성합니다.\n모드에 맞는 양식을 다운로드 후 작성한 PDF를 업로드하세요.",
            style="BodySmall.TLabel",
            justify="left",
        ).grid(row=4, column=0, columnspan=2, pady=(20, 10), sticky="w")

        # 오른쪽 영역: 제공된 이미지를 임베드 (좌/우 1:1 비율로 배치)
        right_holder = ttk.Frame(body, style="Main.TFrame")
        right_holder.grid(row=0, column=1, sticky="nsew", padx=(15, 0))
        right_holder.rowconfigure(0, weight=1)
        right_holder.columnconfigure(0, weight=1)
        right = tk.Label(right_holder, bg=COLOR_BG, borderwidth=0, highlightthickness=0, padx=0, pady=0)
        right.grid(row=0, column=0, sticky="e")
        self._load_right_image(right)

    def download_form(self, mode: str):
        pdf_name = "FontByMe_50.pdf" if mode == "simple" else "FontByMe_220.pdf"
        src = ASSETS_DIR / pdf_name
        if not src.exists():
            messagebox.showerror("에러", f"양식 파일을 찾을 수 없습니다: {src}")
            return
        dest = filedialog.asksaveasfilename(
            title="양식 PDF 저장",
            defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            initialfile=pdf_name,
        )
        if dest:
            try:
                Path(dest).write_bytes(src.read_bytes())
                messagebox.showinfo("완료", f"{dest}에 저장했습니다.")
            except Exception as e:
                messagebox.showerror("에러", f"저장 실패: {e}")

    def start_mode(self, mode: str):
        self.controller.state.mode = mode
        self.controller.show_frame("UploadPage")

    def _load_right_image(self, widget: tk.Label):
        img_path = ASSETS_DIR / "font_by_me_title.png"
        if not img_path.exists():
            # fallback: 작은 캔버스 렌더
            c = tk.Canvas(widget, width=320, height=240, bg=COLOR_BG, highlightthickness=0)
            c.pack(fill="both", expand=True)
            c.create_oval(220, 10, 380, 170, fill="#f4d65c", outline="")
            for y in [30, 90, 150, 210]:
                c.create_line(40, y, 240, y, dash=(6, 4), fill="#b0b0b0")
            c.create_text(140, 80, text="Font", font=("Times New Roman", 42, "bold"), fill="#111111")
            c.create_text(140, 150, text="By Me", font=("Times New Roman", 42, "bold"), fill="#111111")
            return
        # 이미지 로드 및 리사이즈
        try:
            from PIL import Image, ImageTk

            img = Image.open(img_path)
            img = img.resize((380, 320), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            # 우측 끝 정렬
            widget.configure(image=photo, anchor="e")
            widget.image = photo
        except Exception:
            # Pillow가 없으면 PhotoImage로 시도 (PNG만)
            try:
                photo = PhotoImage(file=str(img_path))
                widget.configure(image=photo)
                widget.image = photo
            except Exception:
                widget.configure(text="이미지 로드 실패", bg=COLOR_BG, fg=COLOR_TEXT)


# ----------------------- 업로드 화면 ----------------------- #
class UploadPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.build_ui()

    def build_ui(self):
        self.configure(style="Main.TFrame")
        container = tk.Frame(self, bg=COLOR_BG)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        title_txt = "이 곳에 PDF를 드롭하거나 클릭해서 선택하세요" if HAS_DND else "PDF를 클릭하여 선택하세요"
        tk.Label(container, text=title_txt, font=("Helvetica", 14, "bold"), bg=COLOR_BG, fg=COLOR_TEXT).pack(pady=10)
        drop_area = tk.Frame(container, bg=COLOR_PANEL, width=600, height=320)
        drop_area.pack(pady=10)
        drop_area.pack_propagate(False)
        drop_area.configure(highlightthickness=0, bd=0)
        # 모서리 둥글게: Canvas로 사각형을 그리고 그 위에 컨텐츠 올림
        canvas = tk.Canvas(drop_area, width=600, height=320, bg=COLOR_BG, highlightthickness=0, bd=0)
        canvas.place(x=0, y=0, relwidth=1, relheight=1)
        radius = 30
        w, h = 600, 320
        canvas.create_rectangle(radius, 0, w - radius, h, fill=COLOR_PANEL, outline=COLOR_PANEL)
        canvas.create_rectangle(0, radius, w, h - radius, fill=COLOR_PANEL, outline=COLOR_PANEL)
        canvas.create_oval(0, 0, radius * 2, radius * 2, fill=COLOR_PANEL, outline=COLOR_PANEL)
        canvas.create_oval(w - radius * 2, 0, w, radius * 2, fill=COLOR_PANEL, outline=COLOR_PANEL)
        canvas.create_oval(0, h - radius * 2, radius * 2, h, fill=COLOR_PANEL, outline=COLOR_PANEL)
        canvas.create_oval(w - radius * 2, h - radius * 2, w, h, fill=COLOR_PANEL, outline=COLOR_PANEL)

        content_holder = tk.Frame(drop_area, bg=COLOR_PANEL)
        content_holder.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(content_holder, text="+", font=("Helvetica", 40), bg=COLOR_PANEL, fg=COLOR_TEXT).pack()
        tk.Label(content_holder, text="PDF 선택", font=("Helvetica", 12), bg=COLOR_PANEL, fg=COLOR_TEXT).pack(pady=(6, 0))
        drop_area.bind("<Button-1>", lambda e: self.select_file())
        drop_area.drop_target_register(DND_FILES)
        drop_area.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        if not event.data:
            return
        # tkinterdnd2는 경로가 공백/괄호 포함 시 중괄호로 감싸 전달될 수 있음
        raw = event.data
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        first = raw.split(" ")[0]
        path = Path(first)
        if path.suffix.lower() != ".pdf":
            messagebox.showwarning("알림", "PDF 파일만 지원합니다.")
            return
        self.controller.state.pdf_path = path
        self.controller.start_processing()

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.controller.state.pdf_path = Path(path)
            self.controller.start_processing()


# ----------------------- 로딩 화면 ----------------------- #
class LoadingPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.configure(style="Main.TFrame")
        self.progress = ttk.Progressbar(self, mode="indeterminate", length=600)
        self.msg = ttk.Label(self, text="손글씨 인식 중...", style="Body.TLabel")
        self.msg.pack(pady=30)
        self.progress.pack(fill="x", padx=80)

    def start(self):
        self.progress.start(10)

    def stop(self):
        self.progress.stop()

    def set_message(self, text: str):
        self.msg.config(text=text)


# ----------------------- 결과 화면 ----------------------- #
class ResultPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.build_ui()

    def build_ui(self):
        self.configure(style="Main.TFrame")
        ttk.Label(self, text="폰트 다운로드 하기", style="Bold.TLabel").pack(pady=30)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.btn_ttf = ttk.Button(btn_frame, text="ttf 파일", style="Black.TButton", command=self.download_ttf)
        self.btn_svg = ttk.Button(btn_frame, text="svg 파일", style="Black.TButton", command=self.download_svg)
        self.btn_ttf.grid(row=0, column=0, padx=10)
        self.btn_svg.grid(row=0, column=1, padx=10)
        ttk.Button(self, text="종료하기", style="Gray.TButton", command=self.controller.destroy).pack(pady=20)

    def download_ttf(self):
        if not self.controller.state.output_ttf:
            messagebox.showwarning("없음", "TTF 결과가 없습니다.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".ttf", filetypes=[("TTF", "*.ttf")])
        if dest:
            Path(dest).write_bytes(self.controller.state.output_ttf.read_bytes())
            messagebox.showinfo("완료", "TTF 저장 완료.")

    def download_svg(self):
        if not self.controller.state.output_svg:
            messagebox.showwarning("없음", "SVG 결과가 없습니다.")
            return
        dest = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG", "*.svg")])
        if dest:
            Path(dest).write_bytes(self.controller.state.output_svg.read_bytes())
            messagebox.showinfo("완료", "SVG 저장 완료.")


# ----------------------- 메인 앱 ----------------------- #
class App(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Font By Me")
        self.geometry("700x500")
        self.configure(bg=COLOR_BG)
        self.state = AppState()

        # 스타일 설정
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Main.TFrame", background=COLOR_BG)
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 18, "bold"))
        style.configure("Names.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 11))
        style.configure("Bold.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 12, "bold"))
        style.configure("Body.TLabel", background=COLOR_BG, foreground=COLOR_SUB, font=("Helvetica", 11))
        style.configure("BoldSmall.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Helvetica", 11, "bold"))
        style.configure("BodySmall.TLabel", background=COLOR_BG, foreground=COLOR_SUB, font=("Helvetica", 10))
        style.configure("Gray.TButton", background=COLOR_BTN_GRAY, foreground=COLOR_TEXT, padding=8, relief="flat", borderwidth=0)
        style.map("Gray.TButton", background=[("active", "#b8b8b8")])
        style.configure("GraySmall.TButton", background=COLOR_BTN_GRAY, foreground=COLOR_TEXT, padding=5, relief="flat", borderwidth=0)
        style.map("GraySmall.TButton", background=[("active", "#b8b8b8")])
        style.configure("Black.TButton", background=COLOR_BTN_BLACK, foreground="#ffffff", padding=10, relief="flat", borderwidth=0)
        style.map("Black.TButton", background=[("active", "#222222")])

        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (MainPage, UploadPage, LoadingPage, ResultPage):
            page_name = F.__name__
            frame = F(container, self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("MainPage")

    def show_frame(self, page_name: str):
        frame = self.frames[page_name]
        if page_name == "LoadingPage":
            frame.start()
        elif page_name != "LoadingPage":
            # 다른 페이지로 이동 시 로딩바 정지
            if isinstance(self.frames["LoadingPage"], LoadingPage):
                self.frames["LoadingPage"].stop()
        frame.tkraise()

    def start_processing(self):
        self.show_frame("LoadingPage")
        loader: LoadingPage = self.frames["LoadingPage"]
        loader.set_message("손글씨 인식 중...")
        threading.Thread(target=self._process_workflow, daemon=True).start()

    # 전체 워크플로우 (더미 구현)
    def _process_workflow(self):
        loader: LoadingPage = self.frames["LoadingPage"]
        try:
            # charset 로드
            chars = [c.strip() for c in self.state.charset_file.read_text(encoding="utf-8").splitlines() if c.strip()]
            if not chars:
                raise RuntimeError(f"charset 파일이 비어있습니다: {self.state.charset_file}")
            # 1) PDF → 페이지 이미지 변환 (TODO 실제 구현)
            loader.set_message("PDF 처리 중...")
            images = self._pdf_to_images(self.state.pdf_path, chars, dpi=300)

            # 2) 페이지 중앙 2048x2048 → 256x256 리사이즈 + 라벨 매칭
            loader.set_message("이미지 전처리 중...")
            processed = self._preprocess_images(images)

            # 3) 모델 추론 (TODO 실제 구현)
            loader.set_message("모델 추론/생성 중...")
            glyph_pngs = self._run_model_stub(processed)

            # 4) PNG → SVG (potrace) + SVG → TTF (fontmake) (TODO 실제 구현)
            loader.set_message("SVG/TTF 생성 중...")
            svg_path, ttf_path = self._vectorize_and_build_font(glyph_pngs)

            self.state.output_svg = svg_path
            self.state.output_ttf = ttf_path
            loader.set_message("완료되었습니다.")
            self.show_frame("ResultPage")
        except Exception as e:
            print(f"[ERROR] 처리 중 오류: {e}", file=sys.stderr)
            messagebox.showerror("오류", f"처리 실패: {e}")
            self.show_frame("MainPage")

    # 아래 함수들은 더미/예시 구현입니다. 실제 처리 로직으로 교체하세요.
    def _pdf_to_images(self, pdf_path: Optional[Path], chars: List[str], dpi: int = 300) -> List[Path]:
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("PDF 파일이 없습니다.")
        tmp_dir = OUTPUT_DIR / "tmp_images"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # pdf2image를 사용해 PDF를 이미지로 변환
        try:
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError("pdf2image가 설치되어 있지 않습니다. `pip install pdf2image` 후 재시도하세요.") from e

        # poppler 경로 자동 추정: 환경변수 POPPLER_PATH 우선, macOS brew 기본 경로 fallback
        poppler_env = os.environ.get("POPPLER_PATH")
        poppler_guess = "/opt/homebrew/opt/poppler/bin"
        poppler_path = poppler_env or (poppler_guess if Path(poppler_guess).exists() else None)
        poppler_path = poppler_path or locate_poppler()

        try:
            pages: List[Image.Image] = convert_from_path(
                str(pdf_path), dpi=dpi, poppler_path=poppler_path
            )
        except Exception as e:
            hint = (
                "poppler가 필요합니다. macOS: `brew install poppler`, "
                "Windows: https://github.com/oschwartz10612/poppler-windows/releases 에서 zip 받아 "
                "bin 경로를 POPPLER_PATH 환경변수로 지정하세요."
            )
            raise RuntimeError(f"PDF 처리 실패: {e}\n{hint}") from e
        if not pages:
            raise RuntimeError("PDF에서 페이지를 추출하지 못했습니다.")
        if len(pages) != len(chars):
            print(f"[WARN] 페이지 수({len(pages)})와 charset 길이({len(chars)})가 다릅니다. 짝이 맞는 범위만 저장합니다.")

        out_paths: List[Path] = []
        limit = min(len(pages), len(chars))
        for i in range(limit):
            idx = i + 1
            page = pages[i]
            ch = chars[i]
            if not ch:
                continue
            code_hex = f"{ord(ch):04X}"
            gray = page.convert("L")
            w, h = gray.size
            # 중앙 512 정사각형을 잘라내고 256x256으로 축소
            crop_size = min(512, w, h)
            cx, cy = w // 2, h // 2
            half = crop_size // 2
            left = cx - half
            upper = cy - half
            right = cx + half
            lower = cy + half
            cropped = gray.crop((left, upper, right, lower))
            resized = cropped.resize((256, 256), Image.LANCZOS)
            out_path = tmp_dir / f"{idx:04d}_{code_hex}.png"
            resized.save(out_path)
            out_paths.append(out_path)
        return out_paths

    def _preprocess_images(self, images: List[Path]) -> List[Path]:
        return images

    def _run_model_stub(self, processed_images: List[Path]) -> List[Path]:
        """Fine-tune decoder and generate all glyphs."""
        print(f"[UI] _run_model_stub called with {len(processed_images)} images")
        print(f"[UI] PIPELINE_AVAILABLE = {PIPELINE_AVAILABLE}")

        if not PIPELINE_AVAILABLE:
            print("[WARN] Pipeline not available, returning input images")
            return processed_images

        # Extract characters from processed image filenames (format: 0001_AC00.png)
        chars = []
        for img_path in processed_images:
            # Get hex code from filename
            name = img_path.stem  # e.g., "0001_AC00"
            parts = name.split("_")
            if len(parts) >= 2:
                try:
                    codepoint = int(parts[1], 16)
                    chars.append(chr(codepoint))
                except ValueError:
                    continue

        if not processed_images:
            print("[WARN] No valid images for style extraction")
            return processed_images

        work_dir = OUTPUT_DIR / "pipeline_work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Generate all glyphs using v2 pipeline (font rendering model)
        print(f"[UI] Calling generate_all_glyphs_v2 with {len(processed_images)} style images")
        glyph_dir = work_dir / "glyphs"
        try:
            glyph_paths = generate_all_glyphs_v2(
                style_images=processed_images,
                output_dir=glyph_dir,
            )
            print(f"[UI] Generated {len(glyph_paths)} glyphs")
        except Exception as e:
            print(f"[UI] generate_all_glyphs error: {e}")
            import traceback
            traceback.print_exc()
            return processed_images

        return glyph_paths

    def _vectorize_and_build_font(self, glyph_pngs: List[Path]) -> tuple[Path, Path]:
        """Convert PNGs to SVG and build TTF font."""
        if not PIPELINE_AVAILABLE or not glyph_pngs:
            # Fallback to placeholder
            out_dir = OUTPUT_DIR / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            svg = out_dir / "output.svg"
            ttf = out_dir / "output.ttf"
            svg.write_text("<!-- No glyphs generated -->", encoding="utf-8")
            ttf.write_bytes(b"")
            return svg, ttf

        # Assume glyphs are in pipeline_work/glyphs
        glyph_dir = glyph_pngs[0].parent if glyph_pngs else OUTPUT_DIR / "pipeline_work" / "glyphs"

        font_name = "MyHandwriting"
        ttf_path = OUTPUT_DIR / f"{font_name}.ttf"

        success = create_ttf_font(glyph_dir, ttf_path, font_name)

        if success:
            # Find sample SVG
            svg_dir = glyph_dir / "svg"
            svg_files = list(svg_dir.glob("*.svg")) if svg_dir.exists() else []
            if svg_files:
                svg_path = OUTPUT_DIR / f"{font_name}_sample.svg"
                import shutil
                shutil.copy2(svg_files[0], svg_path)
            else:
                svg_path = OUTPUT_DIR / f"{font_name}_sample.svg"
                svg_path.write_text("<!-- No SVG available -->", encoding="utf-8")

            return svg_path, ttf_path
        else:
            # Fallback
            out_dir = OUTPUT_DIR / "results"
            out_dir.mkdir(parents=True, exist_ok=True)
            svg = out_dir / "output.svg"
            ttf = out_dir / "output.ttf"
            svg.write_text("<!-- Font generation failed -->", encoding="utf-8")
            ttf.write_bytes(b"")
            return svg, ttf


def main():
    run_preflight_checks()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    app = App()
    app.minsize(700, 500)
    app.maxsize(700, 500)
    # 처음에만 앞으로 가져와 포커스, 다른 창 포커스 시 강제로 덮지 않음
    app.after(100, lambda: app.lift())
    app.after(150, lambda: app.focus_force())
    app.after(200, lambda: app.attributes("-topmost", False))
    app.mainloop()


if __name__ == "__main__":
    main()
