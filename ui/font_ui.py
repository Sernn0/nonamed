"""
Tkinter 기반 로컬 UI 예제.

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

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# 프로젝트 루트 기준 기본 경로
ROOT = Path(__file__).resolve().parents[1]
CHARSET_SIMPLE = ROOT / "charset_50.txt"
CHARSET_DETAILED = ROOT / "charset_220.txt"
OUTPUT_DIR = ROOT / "outputs"


# ----------------------- 의존성 체크 ----------------------- #
def check_dependency(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except Exception:
        return False


def run_preflight_checks() -> None:
    print("[INFO] Running preflight checks...")
    # Python 패키지
    py_deps = [
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("fontTools", "fontTools"),
        ("fontmake", "fontmake"),
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
        super().__init__(parent)
        self.controller = controller


# ----------------------- 메인 화면 ----------------------- #
class MainPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.build_ui()

    def build_ui(self):
        # 상단 바
        top = ttk.Frame(self)
        top.pack(fill="x", pady=10, padx=10)
        ttk.Label(top, text="Font By Me", font=("Helvetica", 18, "bold")).pack(side="left")
        ttk.Label(top, text="운여명   황경준   김수인", font=("Helvetica", 11)).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=40, pady=40)

        # 모드 선택
        mode_frame = ttk.Frame(body)
        mode_frame.pack(side="left", anchor="n", padx=20)

        ttk.Label(mode_frame, text="→ 간편 모드", font=("Helvetica", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Button(mode_frame, text="간편 모드 양식 다운로드", command=lambda: self.download_form("simple")).grid(row=0, column=1, padx=6)
        ttk.Button(mode_frame, text="간편 모드로 시작", command=lambda: self.start_mode("simple")).grid(row=0, column=2, padx=6)

        ttk.Label(mode_frame, text="→ 정밀 모드", font=("Helvetica", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Button(mode_frame, text="정밀 모드 양식 다운로드", command=lambda: self.download_form("detailed")).grid(row=1, column=1, padx=6)
        ttk.Button(mode_frame, text="정밀 모드로 시작", command=lambda: self.start_mode("detailed")).grid(row=1, column=2, padx=6)

        ttk.Label(
            mode_frame,
            text="본 앱은 손글씨 PDF를 받아 폰트를 생성합니다.\n모드에 맞는 양식을 다운로드 후 작성한 PDF를 업로드하세요.",
            font=("Helvetica", 11),
            justify="left",
        ).grid(row=2, column=0, columnspan=3, pady=20, sticky="w")

    def download_form(self, mode: str):
        charset = CHARSET_SIMPLE if mode == "simple" else CHARSET_DETAILED
        dest = filedialog.asksaveasfilename(
            title="양식 텍스트 저장",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            initialfile=charset.name,
        )
        if dest:
            try:
                content = charset.read_text(encoding="utf-8")
                Path(dest).write_text(content, encoding="utf-8")
                messagebox.showinfo("완료", f"{dest}에 저장했습니다.")
            except Exception as e:
                messagebox.showerror("에러", f"저장 실패: {e}")

    def start_mode(self, mode: str):
        self.controller.state.mode = mode
        self.controller.show_frame("UploadPage")


# ----------------------- 업로드 화면 ----------------------- #
class UploadPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.build_ui()

    def build_ui(self):
        ttk.Label(self, text="이 곳에 파일을 드롭해주세요", font=("Helvetica", 16)).pack(pady=20)
        drop_area = ttk.Frame(self, padding=40)
        drop_area.pack(pady=30)
        drop_area.configure(relief="solid")
        ttk.Label(drop_area, text="+", font=("Helvetica", 36)).pack()
        ttk.Label(drop_area, text="PDF 선택", font=("Helvetica", 12)).pack(pady=10)
        drop_area.bind("<Button-1>", lambda e: self.select_file())
        # 드래그&드롭은 OS별/추가 패키지가 필요하므로 클릭 선택으로 대체

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if path:
            self.controller.state.pdf_path = Path(path)
            self.controller.start_processing()


# ----------------------- 로딩 화면 ----------------------- #
class LoadingPage(BaseFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.msg = ttk.Label(self, text="손글씨 인식 중...", font=("Helvetica", 14))
        self.msg.pack(pady=20)
        self.progress.pack(fill="x", padx=40)

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
        ttk.Label(self, text="폰트 다운로드 하기", font=("Helvetica", 16)).pack(pady=20)
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        self.btn_ttf = ttk.Button(btn_frame, text="ttf 파일", command=self.download_ttf)
        self.btn_svg = ttk.Button(btn_frame, text="svg 파일", command=self.download_svg)
        self.btn_ttf.grid(row=0, column=0, padx=10)
        self.btn_svg.grid(row=0, column=1, padx=10)

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
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Font By Me")
        self.geometry("1000x700")
        self.state = AppState()

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
            # 1) PDF → 페이지 이미지 변환 (TODO 실제 구현)
            loader.set_message("PDF 처리 중...")
            images = self._pdf_to_images(self.state.pdf_path, dpi=300)

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
    def _pdf_to_images(self, pdf_path: Optional[Path], dpi: int = 300) -> List[Path]:
        if not pdf_path or not pdf_path.exists():
            raise FileNotFoundError("PDF 파일이 없습니다.")
        tmp_dir = OUTPUT_DIR / "tmp_images"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # TODO: pdf2image 등으로 실제 변환
        # 더미: pdf 한 페이지당 빈 PNG 생성
        out = []
        out_path = tmp_dir / "page_001.png"
        from PIL import Image

        Image.new("L", (2048, 2048), 255).save(out_path)
        out.append(out_path)
        return out

    def _preprocess_images(self, images: List[Path]) -> List[Path]:
        # TODO: 실제 크롭(중앙 2048x2048) → 256x256 리사이즈 구현
        return images

    def _run_model_stub(self, processed_images: List[Path]) -> List[Path]:
        # TODO: content/style 인코더+디코더 추론으로 glyph PNG 생성
        return processed_images

    def _vectorize_and_build_font(self, glyph_pngs: List[Path]) -> tuple[Path, Path]:
        # TODO: potrace 호출하여 SVG 변환, fontmake 또는 fontTools로 TTF 빌드
        out_dir = OUTPUT_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        svg = out_dir / "output.svg"
        ttf = out_dir / "output.ttf"
        svg.write_text("<!-- TODO: SVG content -->", encoding="utf-8")
        ttf.write_bytes(b"")  # placeholder
        return svg, ttf


def main():
    run_preflight_checks()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
