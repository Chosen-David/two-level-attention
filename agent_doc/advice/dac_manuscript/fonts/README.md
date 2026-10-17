# Portable paper fonts (2026-10-17)

These are missing-font dependencies for the current IEEEtran/XeLaTeX manuscript, not a claim that Nimbus is the font of every DAC best paper. `Bold` is a face; `BBOLD10` in the Token-Picker reference is a separate math font and is not required here. No fonts were extracted from PDFs. No virtual environment or system font directory is committed.

## Load and verify on the target host

From the repository root, in the same shell used for rendering:

```bash
source agent_doc/advice/dac_manuscript/fonts/env.sh
python3 agent_doc/advice/dac_manuscript/fonts/check_fonts.py --check
python3 agent_doc/advice/dac_manuscript/fonts/proof.py --output /tmp/dac-font-proof
bash agent_doc/advice/dac_manuscript/source/compile_local.sh paper_en paper_zh supplement_en supplement_zh
```

This performs project-local, non-root activation, not a system installation. It creates only the ignored `.font-cache/` beside this directory, sets explicit Fontconfig/TeX/font/Matplotlib/Inkscape paths, and preserves HOME and the caller's executable PATH. `build.sh` uses the same setup, including its system-TeX fallback branch. Missing or altered files, symlinked aliases, wrong internal family/style, or unexpected Fontconfig resolution fail checking. Fontconfig's successful exit alone is not accepted. Arial is an **explicit documented compatibility mapping to Liberation Sans**, never a renamed font file. Existing system Fontconfig remains available for other dependencies.

Prerequisites: Python3, Fontconfig, XeLaTeX/TeX Live with fontspec/unicode-math/IEEEtran, and Poppler utilities for the proof. Python plotting separately needs Matplotlib; SVG conversion needs Inkscape. `.venv-paper` may provide Python packages if actually configured on that host, but activating a venv alone does not install fonts for XeLaTeX or GUI tools. `sources/paper_font_paths.py` registers project fonts per Python process. Existing English DejaVu chart styling and Liberation architecture styling remain intact. The three Chinese renderers now use the actual SC OTF instead of the JP first face returned when Matplotlib reads a TTC filename.

For GUI applications outside the activated shell, optionally copy this complete directory (fonts **and licenses/source archive**) to a user-owned font directory, e.g. `${XDG_DATA_HOME:-$HOME/.local/share}/fonts/dac-paper`, and run `fc-cache -f` on that directory. Do not overwrite a different existing bundle. Restart the GUI. No remote host installation or GUI/deck export is claimed by this repository publication.

## Exact dependencies / licenses

`MANIFEST.json` records every redistributed font/license/source file's SHA256, original internal family/style/PostScript identity and acquisition route.

- `urw/`: eight untouched Nimbus Roman/Sans OTFs from Artifex commit `3c0ba3b5687632dfc66526544a4e811fe0ec0cd9`. **AGPL-3 with PDF/PostScript embedding exception**, not OFL. Original LICENSE, full COPYING and README retained; `sources/urw-base35-3c0ba3b.tar.gz` supplies the complete unmodified upstream source snapshot (including Type1/TTF/OTF/metrics), alongside the selected OTFs. Font software retains its own license; the embedding exception covers document PDFs.
- `noto/`: official Sans2.004/Serif2.003, commit `f8d157532fbfaeda587e826d4cd5b21a49186f7c`, regular/bold **SC-specific OTF** files from each family's `OTF/SimplifiedChinese/`. OFL-1.1; original family licenses and embedded copyright retained. Four OTFs replace the need for four large TTCs. Chinese main/supplement changes are font-loading declarations only. No OTF is renamed to a TTC.
- `liberation/`: four unmodified installed Liberation Sans2.1.5 TTFs, OFL-1.1, with full distribution copyright/license and reserved-name notices. This preserves the architecture family's intended metrics; it does not bundle Microsoft Arial.
- `lm-math/`: unmodified Latin Modern Math1.959 from the existing TeX Live distribution, GUST Font License / LPPL1.3c-or-later; original license, README and manifest included. The official CTAN download returned HTTP403; no fresh upstream-byte comparison is claimed. Installed original hash is pinned. This is not a font extracted from a paper.

No LibreOffice/PPTX rerender, new font-family redesign, figure regeneration, scientific-text change, or authoritative manuscript PDF publication is included.

## Verification and limitations

Local exact-byte/face checks and 7 × 3.4 inch proof pass: all 17 requested font faces including math embedded; no missing characters, overflow, or shape warnings in that independent proof. Visual inspection confirms regular/bold/italic distinctions, SC text, arrows and math. English/Chinese Matplotlib3.10.8 SVG→PDF smoke runs passed using the actual local interpreter (not a claimed `.venv-paper`). Inkscape initially emitted read-only profile warnings; project profile path is now set.

Four manuscripts were built **in scratch**, not committed: main EN/ZH 6 pages each, supplements 8 each, zero missing-character/overfull diagnostics. Main EN retains 6 font warnings; main ZH retains 8: early IEEEtran TU/ptm initialization plus actual unsupported small-caps and Chinese italic shapes. These warnings are not suppressed or called zero-substitution. Supplements had zero font warnings. The OTF main ZH embeds actual `NotoSerifCJKsc`, whereas a preliminary TTC proof's PDF names showed JP; no claim is made that every historical TTC glyph was wrong. This batch ensures requested installed faces and reports remaining document shape policy for the manuscript owner. Existing embedded figure fonts remain unchanged. Remote rebuilding and page/layout approval are still host actions.
