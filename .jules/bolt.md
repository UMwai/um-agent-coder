# Bolt's Journal

## 2024-05-22 - [Initialization]
**Learning:** Journal initialized.
**Action:** Record critical learnings here.

## 2024-05-22 - [Lazy Regex Parsing]
**Learning:** `re.findall` eagerly loads all matches into memory, which is inefficient for large inputs when only a subset is needed. `re.finditer` with `itertools.islice` is significantly faster (~378x on 10k items) and uses less memory.
**Action:** Prefer `re.finditer` + `itertools.islice` for parsing large text data when only top N results are needed. Pre-compile regex patterns for repeated use.
