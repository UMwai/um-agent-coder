# Palette's UX Journal

## 2024-05-22 - CLI Spinners
**Learning:** Python CLIs need explicit handling of `\r` and `\033[K` for clean spinner updates, unlike web UIs where DOM updates handle this.
**Action:** When implementing spinners or progress bars, always ensure the previous line is cleared before writing the new state to avoid "ghost" characters.
