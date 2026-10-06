#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_dupes.py - поиск дублей картинок (JPEG / PNG / WebP) в папке.

Сравнивает картинки по содержимому, а не по байтам, поэтому находит дубли
с разными именами, разрешением, форматом и качеством сжатия.

Каждая найденная группа дублей перемещается в свою подпапку:
    _dupes\\001, _dupes\\002, ...

Запуск (в папке с картинками):
    python find_dupes.py                 текущая папка, группы дублей - по подпапкам
    python find_dupes.py D:\\Photos       указанная папка
    python find_dupes.py -n              только показать список, ничего не перемещать
    python find_dupes.py -k              лучшую картинку группы оставить на месте,
                                         переместить только её копии
    python find_dupes.py -d Дубли        другое имя папки для дублей (вместо _dupes)
    python find_dupes.py -r              вместе с подпапками
    python find_dupes.py -t 6            строже (меньше ложных срабатываний)
    python find_dupes.py -t 20           мягче (ловит более отличающиеся копии)
    python find_dupes.py -o dupes.txt    дополнительно сохранить отчёт в файл

Требуется: Python 3.8+ и Pillow  (pip install pillow)

Скрипт ничего не удаляет и не перезаписывает: файлы только перемещаются.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("Не найдена библиотека Pillow. Установите её:\n    pip install pillow")

EXTENSIONS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp"}
ASPECT_TOLERANCE = 0.04   # допустимая разница пропорций сторон (4 %)
THUMB_TOLERANCE = 12.0    # допустимая средняя разница яркости миниатюр 8x8 (0..255)

Image.MAX_IMAGE_PIXELS = None  # не ругаться на очень большие картинки

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # старые версии Pillow
    RESAMPLE = Image.LANCZOS

popcount = getattr(int, "bit_count", None) or (lambda x: bin(x).count("1"))


def fingerprint(path):
    """Возвращает (hash128, thumb8x8, ширина, высота) или None, если файл не читается."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            try:
                if img.getexif().get(0x0112) in (5, 6, 7, 8):  # повёрнута на 90°
                    width, height = height, width
            except Exception:
                pass

            img.draft("RGB", (256, 256))  # ускоряет чтение больших JPEG
            img = ImageOps.exif_transpose(img)

            # прозрачность кладём на белый фон
            if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
                img = img.convert("RGBA")
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(background, img)

            gray = img.convert("L")

            bits = 0
            # горизонтальный dHash: 8 строк по 8 сравнений соседних пикселей
            a = gray.resize((9, 8), RESAMPLE).tobytes()
            for y in range(8):
                for x in range(8):
                    bits = (bits << 1) | (a[y * 9 + x] > a[y * 9 + x + 1])
            # вертикальный dHash
            b = gray.resize((8, 9), RESAMPLE).tobytes()
            for y in range(8):
                for x in range(8):
                    bits = (bits << 1) | (b[y * 8 + x] > b[(y + 1) * 8 + x])

            thumb = gray.resize((8, 8), RESAMPLE).tobytes()
            return bits, thumb, width, height
    except Exception:
        return None


def thumbs_close(t1, t2):
    return sum(abs(p - q) for p, q in zip(t1, t2)) / len(t1) <= THUMB_TOLERANCE


def human_size(n):
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024


def unique_path(path):
    """Если файл с таким именем уже есть, добавляет к имени (2), (3), ..."""
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def next_group_number(dest):
    """Номер для следующей подпапки: продолжает нумерацию уже существующих."""
    if not dest.is_dir():
        return 1
    numbers = [int(p.name) for p in dest.iterdir() if p.is_dir() and p.name.isdigit()]
    return max(numbers, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description="Поиск дублей картинок (JPEG/PNG/WebP).")
    parser.add_argument("folder", nargs="?", default=".", help="папка (по умолчанию текущая)")
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="только показать дубли, ничего не перемещать")
    parser.add_argument("-k", "--keep-best", action="store_true",
                        help="самую большую картинку группы оставить на месте, переместить только копии")
    parser.add_argument("-d", "--dest", default="_dupes",
                        help="папка для дублей (по умолчанию _dupes)")
    parser.add_argument("-r", "--recursive", action="store_true", help="искать и в подпапках")
    parser.add_argument("-t", "--threshold", type=int, default=12,
                        help="порог различия хэшей, 0..128 (по умолчанию 12; меньше = строже)")
    parser.add_argument("-o", "--output", help="дополнительно сохранить отчёт в файл")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    root = Path(args.folder).resolve()
    if not root.is_dir():
        sys.exit(f"Папка не найдена: {root}")

    dest = (root / args.dest).resolve()

    def show(path):
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    walker = root.rglob("*") if args.recursive else root.iterdir()
    files = sorted(p for p in walker
                   if p.is_file() and p.suffix.lower() in EXTENSIONS
                   and dest not in p.parents)  # уже разложенные дубли не трогаем
    if not files:
        sys.exit(f"В папке {root} нет картинок JPEG/PNG/WebP.")

    # --- 1. считаем отпечатки ---
    items, broken = [], []
    workers = min(8, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (path, fp) in enumerate(zip(files, pool.map(fingerprint, files)), 1):
            if fp is None:
                broken.append(path)
            else:
                bits, thumb, w, h = fp
                items.append((w / h, bits, thumb, w, h, path))
            if done % 25 == 0 or done == len(files):
                print(f"\rОбработано {done} из {len(files)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    # --- 2. сравниваем попарно (только картинки с близкими пропорциями) ---
    items.sort(key=lambda it: it[0])
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ratio_i, bits_i, thumb_i = items[i][0], items[i][1], items[i][2]
        limit = ratio_i * (1 + ASPECT_TOLERANCE)
        for j in range(i + 1, n):
            if items[j][0] > limit:
                break
            if popcount(bits_i ^ items[j][1]) <= args.threshold and thumbs_close(thumb_i, items[j][2]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    groups = [g for g in groups.values() if len(g) > 1]
    # внутри группы - от самой большой картинки к самой маленькой
    for g in groups:
        g.sort(key=lambda it: (-(it[3] * it[4]), -it[5].stat().st_size, str(it[5]).lower()))
    groups.sort(key=lambda g: str(g[0][5]).lower())

    # --- 3. перемещение и отчёт ---
    lines = []
    moved = failed = 0
    if not groups:
        lines.append("Дублей не найдено.")
    for num, g in enumerate(groups, next_group_number(dest)):
        target = dest / f"{num:03d}"
        header = f"Группа {num} ({len(g)} шт.)"
        lines.append(header + (":" if args.dry_run else f" -> {show(target)}:"))
        for idx, (_, _, _, w, h, path) in enumerate(g):
            line = f"    {show(path)}    [{w}x{h}, {human_size(path.stat().st_size)}]"
            if args.dry_run:
                pass
            elif args.keep_best and idx == 0:
                line += "    (оставлена на месте)"
            else:
                try:
                    target.mkdir(parents=True, exist_ok=True)
                    new_path = unique_path(target / path.name)
                    shutil.move(str(path), str(new_path))
                    moved += 1
                    if new_path.name != path.name:
                        line += f"    (переименована в {new_path.name})"
                except OSError as err:
                    failed += 1
                    line += f"    НЕ ПЕРЕМЕЩЕНА: {err}"
            lines.append(line)
        lines.append("")
    if groups:
        extra = sum(len(g) - 1 for g in groups)
        lines.append(f"Всего картинок: {len(files)}. Групп дублей: {len(groups)}, лишних копий: {extra}.")
        if args.dry_run:
            lines.append("Режим просмотра (-n): файлы не перемещались.")
        else:
            lines.append(f"Перемещено файлов: {moved}, папка: {dest}")
            if failed:
                lines.append(f"Не удалось переместить: {failed} (см. пометки выше).")
    if broken:
        lines.append("")
        lines.append(f"Не удалось прочитать ({len(broken)}):")
        lines.extend(f"    {show(p)}" for p in broken)

    report = "\n".join(lines)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
