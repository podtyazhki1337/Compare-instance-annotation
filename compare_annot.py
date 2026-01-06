#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instance mask comparison - ПРАВИЛЬНАЯ ВЕРСИЯ

ИСПРАВЛЕНИЯ:
✓ Слайдер в пределах bbox объекта с реальными срезами
✓ Instance mask с background=0 для napari
✓ Автоматическое добавление непарных объектов
✓ Индикация объектов только из A или только из B

Управление:
  1/A: выбрать A
  2/B: выбрать B
  3/M: merge A+B
  4/N: пропустить (не добавлять)
  U: undo
  Q: save
  ←/→: navigation
"""
import argparse, sys, os, gc, time
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from scipy.ndimage import find_objects

try:
    import tifffile
except:
    tifffile = None
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


def load_volume(path):
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        if not tifffile:
            raise ImportError("tifffile required")
        try:
            arr = tifffile.memmap(path, mode='r')
            return np.array(arr) if isinstance(arr, np.memmap) else arr
        except:
            return tifffile.imread(path)
    raise ValueError(f"Unsupported: {ext}")


def save_instance_mask(path, vol):
    if not tifffile:
        raise ImportError("tifffile required")
    if path.suffix.lower() not in (".tif", ".tiff"):
        path = path.with_suffix(".tif")
    path.parent.mkdir(parents=True, exist_ok=True)
    vol_out = vol.astype(np.uint16)
    unique_ids = np.unique(vol_out)
    if unique_ids[0] != 0:
        print(f"[WARNING] Background={unique_ids[0]} ≠ 0")
    tifffile.imwrite(str(path), vol_out, bigtiff=(vol_out.nbytes >= int(3.8 * 1024 ** 3)),
                     photometric='minisblack', compression='zlib', compressionargs={'level': 6},
                     metadata={'axes': 'ZYX', 'mode': 'composite'})
    print(f"[SAVE] {path.name} ({path.stat().st_size / 1e6:.1f}MB, {len(unique_ids) - 1}obj, bg={unique_ids[0]})")


def normalize_slice(img2d, p_low=1.0, p_high=99.0):
    img = img2d.astype(np.float32)
    if img.size == 0:
        return np.zeros_like(img)
    lo, hi = np.percentile(img, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((img - lo) / (hi - lo), 0, 1)


def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def find_matching_pairs(mask_a, mask_b, iou_threshold=0.1):
    ids_a = sorted([int(x) for x in np.unique(mask_a) if x > 0])
    ids_b = sorted([int(x) for x in np.unique(mask_b) if x > 0])
    pairs, matched_b = [], set()
    for id_a in ids_a:
        mask_a_obj = (mask_a == id_a)
        best_iou, best_id_b = 0.0, None
        for id_b in ids_b:
            if id_b in matched_b:
                continue
            iou = compute_iou(mask_a_obj, mask_b == id_b)
            if iou > best_iou and iou >= iou_threshold:
                best_iou, best_id_b = iou, id_b
        if best_id_b:
            matched_b.add(best_id_b)
            pairs.append((id_a, best_id_b, best_iou))
        else:
            pairs.append((id_a, None, 0.0))
    for id_b in ids_b:
        if id_b not in matched_b:
            pairs.append((None, id_b, 0.0))
    return pairs


def get_bbox(mask, obj_id, margin=10):
    obj_mask = (mask == obj_id)
    slices = find_objects(obj_mask.astype(np.uint8))[0]
    if slices is None:
        return slice(0, 1), slice(0, 1), slice(0, 1)
    zsl, ysl, xsl = slices
    shape = mask.shape
    return (slice(max(0, zsl.start - margin), min(shape[0], zsl.stop + margin)),
            slice(max(0, ysl.start - margin), min(shape[1], ysl.stop + margin)),
            slice(max(0, xsl.start - margin), min(shape[2], xsl.stop + margin)))


def best_slice_in_bbox(mask, obj_id, bbox):
    zb, yb, xb = bbox
    obj_mask = (mask[zb, yb, xb] == obj_id)
    z_sums = obj_mask.sum(axis=(1, 2))
    return int(np.argmax(z_sums)) if z_sums.sum() > 0 else 0


def intersect_names(dir_img, dir_a, dir_b, exts=(".tif", ".tiff", ".npy")):
    names_img = {p.name for p in dir_img.iterdir() if p.is_file() and p.suffix.lower() in exts}
    names_a = {p.name for p in dir_a.iterdir() if p.is_file() and p.suffix.lower() in exts}
    names_b = {p.name for p in dir_b.iterdir() if p.is_file() and p.suffix.lower() in exts}
    return sorted(names_img & names_a & names_b)


class ReviewState:
    def __init__(self, n_pairs):
        self.pair_idx = 0
        self.n_pairs = n_pairs
        self.z_local = 0
        self.chosen_a = 0
        self.chosen_b = 0
        self.chosen_merge = 0
        self.chosen_none = 0  # Добавляем счетчик пропущенных
        self.history = []
        self.updating_slider = False


class InstanceMaskReviewer:
    def __init__(self, img_path, a_path, b_path, out_mask_path, margin, iou_threshold):
        print(f"[LOAD] {img_path.name}...")
        t0 = time.time()
        self.img = load_volume(str(img_path))
        self.mask_a = load_volume(str(a_path))
        self.mask_b = load_volume(str(b_path))
        print(f"[LOAD] {time.time() - t0:.1f}s")
        if self.img.shape != self.mask_a.shape or self.img.shape != self.mask_b.shape:
            raise ValueError("Shape mismatch")
        self.img_path = img_path
        self.out_mask_path = out_mask_path
        self.margin = margin
        print("[PAIR] Finding...")
        self.pairs = find_matching_pairs(self.mask_a, self.mask_b, iou_threshold)
        if len(self.pairs) == 0:
            print("[INFO] No objects")
            save_instance_mask(out_mask_path, self.mask_a)
            self.stats = {"pairs": 0, "chosen_a": 0, "chosen_b": 0, "chosen_merge": 0, "chosen_none": 0}
            return
        print(f"[PAIR] Found {len(self.pairs)}")
        self.result = np.zeros_like(self.mask_a, dtype=np.uint16)
        self.next_id = 1
        self.state = ReviewState(len(self.pairs))
        self.cache_bbox_a = None
        self.cache_bbox_b = None
        self.z_slices_available = [0]  # Инициализация
        self.fig = None
        self.axes = {}
        self.images = {}
        self.widgets = {}
        self.setup_ui()

    def get_current_pair(self):
        if self.state.pair_idx >= len(self.pairs):
            return None, None, 0.0
        return self.pairs[self.state.pair_idx]

    def update_bbox_cache(self):
        """Обновляет bbox и находит РЕАЛЬНЫЕ срезы с объектом."""
        id_a, id_b, _ = self.get_current_pair()

        # Bbox
        self.cache_bbox_a = get_bbox(self.mask_a, id_a, self.margin) if id_a else None
        self.cache_bbox_b = get_bbox(self.mask_b, id_b, self.margin) if id_b else None

        # Находим РЕАЛЬНЫЕ Z-срезы где есть объект
        z_slices_with_obj = []

        if id_a:
            # Проверяем каждый срез в bbox A
            zb, _, _ = self.cache_bbox_a
            for z in range(zb.start, zb.stop):
                if (self.mask_a[z] == id_a).any():
                    z_slices_with_obj.append(z)

        if id_b and not z_slices_with_obj:
            # Если A пустой, берём B
            zb, _, _ = self.cache_bbox_b
            for z in range(zb.start, zb.stop):
                if (self.mask_b[z] == id_b).any():
                    z_slices_with_obj.append(z)

        # Если ничего не нашли - берём весь диапазон bbox
        if not z_slices_with_obj:
            if self.cache_bbox_a:
                zb = self.cache_bbox_a[0]
                z_slices_with_obj = list(range(zb.start, zb.stop))
            elif self.cache_bbox_b:
                zb = self.cache_bbox_b[0]
                z_slices_with_obj = list(range(zb.start, zb.stop))
            else:
                z_slices_with_obj = [0]

        self.z_slices_available = sorted(set(z_slices_with_obj))

    def local_to_global_z(self, z_local):
        """Преобразует локальный индекс (0,1,2...) в глобальный Z."""
        if z_local < 0 or z_local >= len(self.z_slices_available):
            return self.z_slices_available[0]
        return self.z_slices_available[z_local]

    def render_image(self, img_crop):
        if img_crop.size == 0:
            return np.full((50, 50, 3), 0.5, dtype=np.float32)
        img_norm = normalize_slice(img_crop)
        return np.stack([img_norm] * 3, axis=-1)

    def render_mask_overlay(self, img_crop, mask_crop, obj_id, color):
        if img_crop.size == 0:
            return np.full((50, 50, 3), 0.5, dtype=np.float32)
        img_norm = normalize_slice(img_crop)
        rgb = np.stack([img_norm] * 3, axis=-1)
        if mask_crop is not None and obj_id is not None:
            mask_obj = (mask_crop == obj_id)
            if color == 'red':
                rgb[mask_obj, 0] = np.clip(rgb[mask_obj, 0] + 0.7, 0, 1)
            elif color == 'green':
                rgb[mask_obj, 1] = np.clip(rgb[mask_obj, 1] + 0.7, 0, 1)
        return rgb

    def recreate_slider_if_needed(self):
        """Пересоздаёт слайдер если изменилось количество срезов."""
        n_z = len(self.z_slices_available)

        # Проверяем нужно ли пересоздавать
        if hasattr(self, 'z_slider') and hasattr(self, 'last_n_z'):
            if self.last_n_z == n_z:
                # Просто обновляем значение
                self.state.updating_slider = True
                try:
                    self.z_slider.set_val(self.state.z_local)
                finally:
                    self.state.updating_slider = False
                return

        # Нужно пересоздать слайдер
        if hasattr(self, 'z_slider'):
            # Удаляем старый
            self.z_slider.ax.remove()
            del self.z_slider

        # Создаём новый
        ax_slider = plt.axes([0.15, 0.12, 0.7, 0.03])
        self.z_slider = Slider(
            ax_slider,
            f'Z({n_z}slices)',
            0,
            max(0, n_z - 1),
            valinit=min(self.state.z_local, n_z - 1),
            valstep=1
        )
        self.z_slider.on_changed(self.on_slider_change)
        self.last_n_z = n_z

        # Обновляем z_local если вышли за пределы
        if self.state.z_local >= n_z:
            self.state.z_local = n_z - 1

    def update_display(self):
        self.update_bbox_cache()
        id_a, id_b, iou = self.get_current_pair()
        z_global = self.local_to_global_z(self.state.z_local)

        if id_a is None and id_b is None:
            return

        # Crops для A
        if id_a and self.cache_bbox_a:
            zb_a, yb_a, xb_a = self.cache_bbox_a
            z_a = z_global

            img_crop_a = self.img[z_a, yb_a, xb_a]
            mask_crop_a = self.mask_a[z_a, yb_a, xb_a]
            vox_a = (self.mask_a == id_a).sum()
            n_slices_a = sum(1 for z in range(zb_a.start, zb_a.stop) if (self.mask_a[z] == id_a).any())
            range_a = f"z:{zb_a.start}-{zb_a.stop - 1}({n_slices_a}slices)"
        else:
            img_crop_a = np.zeros((50, 50))
            mask_crop_a = None
            vox_a = 0
            range_a = "N/A"

        # Crops для B
        if id_b and self.cache_bbox_b:
            zb_b, yb_b, xb_b = self.cache_bbox_b
            z_b = z_global

            img_crop_b = self.img[z_b, yb_b, xb_b]
            mask_crop_b = self.mask_b[z_b, yb_b, xb_b]
            vox_b = (self.mask_b == id_b).sum()
            n_slices_b = sum(1 for z in range(zb_b.start, zb_b.stop) if (self.mask_b[z] == id_b).any())
            range_b = f"z:{zb_b.start}-{zb_b.stop - 1}({n_slices_b}slices)"
        else:
            img_crop_b = np.zeros((50, 50))
            mask_crop_b = None
            vox_b = 0
            range_b = "N/A"

        # Рендер
        panel_img_a = self.render_image(img_crop_a)
        panel_mask_a = self.render_mask_overlay(img_crop_a, mask_crop_a, id_a, 'red')
        panel_img_b = self.render_image(img_crop_b)
        panel_mask_b = self.render_mask_overlay(img_crop_b, mask_crop_b, id_b, 'green')

        # Обновляем images
        self.images['img_a'].set_data(panel_img_a)
        self.images['img_a'].set_extent([0, panel_img_a.shape[1], panel_img_a.shape[0], 0])
        self.images['mask_a'].set_data(panel_mask_a)
        self.images['mask_a'].set_extent([0, panel_mask_a.shape[1], panel_mask_a.shape[0], 0])
        self.images['img_b'].set_data(panel_img_b)
        self.images['img_b'].set_extent([0, panel_img_b.shape[1], panel_img_b.shape[0], 0])
        self.images['mask_b'].set_data(panel_mask_b)
        self.images['mask_b'].set_extent([0, panel_mask_b.shape[1], panel_mask_b.shape[0], 0])

        for ax_name, panel in [('img_a', panel_img_a), ('mask_a', panel_mask_a),
                               ('img_b', panel_img_b), ('mask_b', panel_mask_b)]:
            self.axes[ax_name].set_xlim(0, panel.shape[1])
            self.axes[ax_name].set_ylim(panel.shape[0], 0)

        # Заголовки - добавляем индикацию непарных объектов
        if id_a and not id_b:
            title_img_a = "Image A ⚠️UNPAIRED" if id_a else "A:None"
            title_mask_a = f"⚠️ONLY in A|ID={id_a}|{vox_a}vox|{range_a}" if id_a else "A:None"
        else:
            title_img_a = "Image A" if id_a else "A:None"
            title_mask_a = f"Mask A|ID={id_a}|{vox_a}vox|{range_a}" if id_a else "A:None"

        if id_b and not id_a:
            title_img_b = "Image B ⚠️UNPAIRED" if id_b else "B:None"
            title_mask_b = f"⚠️ONLY in B|ID={id_b}|{vox_b}vox|{range_b}" if id_b else "B:None"
        else:
            title_img_b = "Image B" if id_b else "B:None"
            title_mask_b = f"Mask B|ID={id_b}|{vox_b}vox|{range_b}" if id_b else "B:None"

        self.axes['img_a'].set_title(title_img_a, fontsize=9)
        self.axes['mask_a'].set_title(title_mask_a, fontsize=8, color='red', fontweight='bold')
        self.axes['img_b'].set_title(title_img_b, fontsize=9)
        self.axes['mask_b'].set_title(title_mask_b, fontsize=8, color='green', fontweight='bold')

        n_z_available = len(self.z_slices_available)
        self.fig.suptitle(f"{self.img_path.name}|Pair{self.state.pair_idx + 1}/{len(self.pairs)}|"
                          f"IoU={iou:.3f}|z_local={self.state.z_local}/{n_z_available - 1}(glob={z_global})|"
                          f"A:{self.state.chosen_a} B:{self.state.chosen_b} M:{self.state.chosen_merge} N:{self.state.chosen_none}",
                          fontsize=9, fontweight='bold')

        # КРИТИЧНО: Пересоздаём слайдер если нужно
        self.recreate_slider_if_needed()

        self.fig.canvas.draw_idle()

    def setup_ui(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.patch.set_facecolor('#2e2e2e')
        gs = self.fig.add_gridspec(3, 4, left=0.05, right=0.95, top=0.92, bottom=0.18,
                                   wspace=0.15, hspace=0.20)
        self.axes['img_a'] = self.fig.add_subplot(gs[0, 0:2])
        self.axes['mask_a'] = self.fig.add_subplot(gs[1, 0:2])
        self.axes['img_b'] = self.fig.add_subplot(gs[0, 2:4])
        self.axes['mask_b'] = self.fig.add_subplot(gs[1, 2:4])
        for ax in self.axes.values():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        empty = np.full((100, 100, 3), 0.5)
        self.images['img_a'] = self.axes['img_a'].imshow(empty, aspect='equal', interpolation='nearest')
        self.images['mask_a'] = self.axes['mask_a'].imshow(empty, aspect='equal', interpolation='nearest')
        self.images['img_b'] = self.axes['img_b'].imshow(empty, aspect='equal', interpolation='nearest')
        self.images['mask_b'] = self.axes['mask_b'].imshow(empty, aspect='equal', interpolation='nearest')
        # Слайдер - создаём с помощью новой функции
        self.update_bbox_cache()

        id_a, _, _ = self.get_current_pair()
        if id_a and self.cache_bbox_a:
            # Находим лучший срез ВНУТРИ списка доступных
            best_z_global = self.z_slices_available[0] + best_slice_in_bbox(
                self.mask_a, id_a, self.cache_bbox_a
            )
            # Преобразуем в локальный индекс
            try:
                self.state.z_local = self.z_slices_available.index(best_z_global)
            except ValueError:
                self.state.z_local = 0
        else:
            self.state.z_local = 0

        self.last_n_z = -1  # Принудительно создаём слайдер
        self.recreate_slider_if_needed()
        btn_h, btn_w, y, gap = 0.05, 0.09, 0.05, 0.012
        x = 0.05
        self.widgets['btn_a'] = Button(plt.axes([x, y, btn_w, btn_h]), '1.A', color='lightcoral')
        self.widgets['btn_a'].on_clicked(lambda e: self.choose_mask('A'))
        x += btn_w + gap
        self.widgets['btn_b'] = Button(plt.axes([x, y, btn_w, btn_h]), '2.B', color='lightgreen')
        self.widgets['btn_b'].on_clicked(lambda e: self.choose_mask('B'))
        x += btn_w + gap
        self.widgets['btn_merge'] = Button(plt.axes([x, y, btn_w, btn_h]), '3.M', color='lightyellow')
        self.widgets['btn_merge'].on_clicked(lambda e: self.choose_mask('Merge'))
        x += btn_w + gap
        self.widgets['btn_none'] = Button(plt.axes([x, y, btn_w, btn_h]), '4.None', color='lightgray')
        self.widgets['btn_none'].on_clicked(lambda e: self.choose_mask('None'))
        x += btn_w + gap
        self.widgets['btn_undo'] = Button(plt.axes([x, y, btn_w, btn_h]), 'U.Undo')
        self.widgets['btn_undo'].on_clicked(lambda e: self.undo())
        x += btn_w + gap + 0.05
        self.widgets['btn_prev'] = Button(plt.axes([x, y, btn_w / 2, btn_h]), '←')
        self.widgets['btn_prev'].on_clicked(lambda e: self.move_prev())
        x += btn_w / 2 + 0.005
        self.widgets['btn_next'] = Button(plt.axes([x, y, btn_w / 2, btn_h]), '→')
        self.widgets['btn_next'].on_clicked(lambda e: self.move_next())
        x += btn_w / 2 + gap + 0.05
        self.widgets['btn_save'] = Button(plt.axes([x, y, btn_w, btn_h]), 'Q.Save', color='orange')
        self.widgets['btn_save'].on_clicked(lambda e: self.save_and_quit())
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.update_display()

    def on_slider_change(self, val):
        if self.state.updating_slider:
            return
        self.state.z_local = int(val)
        self.update_display()

    def choose_mask(self, which):
        id_a, id_b, _ = self.get_current_pair()

        self.state.history.append((self.state.pair_idx, self.result.copy(), self.next_id))

        if which == 'A' and id_a:
            source_mask = (self.mask_a == id_a)
            self.result[source_mask] = self.next_id
            self.state.chosen_a += 1
            self.next_id += 1
        elif which == 'B' and id_b:
            source_mask = (self.mask_b == id_b)
            self.result[source_mask] = self.next_id
            self.state.chosen_b += 1
            self.next_id += 1
        elif which == 'Merge':
            source_mask = np.zeros_like(self.mask_a, dtype=bool)
            if id_a:
                source_mask = np.logical_or(source_mask, self.mask_a == id_a)
            if id_b:
                source_mask = np.logical_or(source_mask, self.mask_b == id_b)
            self.result[source_mask] = self.next_id
            self.state.chosen_merge += 1
            self.next_id += 1
        elif which == 'None':
            self.state.chosen_none += 1
            # ничего не добавляем - просто счетчик

        self.save_current()
        self.move_next()

    def move_next(self):
        if self.state.pair_idx < len(self.pairs) - 1:
            self.state.pair_idx += 1
            self.update_bbox_cache()

            # Находим лучший срез для нового объекта
            id_a, _, _ = self.get_current_pair()
            if id_a and self.cache_bbox_a:
                best_z_local_in_bbox = best_slice_in_bbox(self.mask_a, id_a, self.cache_bbox_a)
                # Конвертируем в глобальный
                zb = self.cache_bbox_a[0]
                best_z_global = zb.start + best_z_local_in_bbox
                # Находим в нашем списке доступных срезов
                try:
                    self.state.z_local = self.z_slices_available.index(best_z_global)
                except ValueError:
                    self.state.z_local = 0
            else:
                self.state.z_local = 0

            self.update_display()
        else:
            print("[INFO] Done")
            self.save_and_quit()

    def move_prev(self):
        if self.state.pair_idx > 0:
            self.state.pair_idx -= 1
            self.update_bbox_cache()

            id_a, _, _ = self.get_current_pair()
            if id_a and self.cache_bbox_a:
                best_z_local_in_bbox = best_slice_in_bbox(self.mask_a, id_a, self.cache_bbox_a)
                zb = self.cache_bbox_a[0]
                best_z_global = zb.start + best_z_local_in_bbox
                try:
                    self.state.z_local = self.z_slices_available.index(best_z_global)
                except ValueError:
                    self.state.z_local = 0
            else:
                self.state.z_local = 0

            self.update_display()

    def undo(self):
        if not self.state.history:
            print("[INFO] No history")
            return
        idx, old_result, old_next_id = self.state.history.pop()
        self.result = old_result
        self.next_id = old_next_id
        self.state.pair_idx = idx
        self.save_current()
        self.update_display()

    def save_current(self):
        try:
            save_instance_mask(self.out_mask_path, self.result)
        except Exception as e:
            print(f"[ERROR] Save: {e}")

    def save_and_quit(self):
        # Автоматически добавляем все непроверенные объекты
        print("[INFO] Adding unpaired objects...", flush=True)
        for idx in range(self.state.pair_idx + 1, len(self.pairs)):
            id_a, id_b, _ = self.pairs[idx]

            # Если объект есть только в A - добавляем его
            if id_a and not id_b:
                source_mask = (self.mask_a == id_a)
                self.result[source_mask] = self.next_id
                self.next_id += 1
                print(f"  Added A obj {id_a}", flush=True)

            # Если объект есть только в B - добавляем его
            elif id_b and not id_a:
                source_mask = (self.mask_b == id_b)
                self.result[source_mask] = self.next_id
                self.next_id += 1
                print(f"  Added B obj {id_b}", flush=True)

            # Если есть оба (пара) - добавляем тот у которого больше IoU или объём
            elif id_a and id_b:
                # Берём объект с большим объёмом
                vox_a = (self.mask_a == id_a).sum()
                vox_b = (self.mask_b == id_b).sum()
                if vox_a >= vox_b:
                    source_mask = (self.mask_a == id_a)
                    self.result[source_mask] = self.next_id
                    print(f"  Added A obj {id_a} (vox={vox_a})", flush=True)
                else:
                    source_mask = (self.mask_b == id_b)
                    self.result[source_mask] = self.next_id
                    print(f"  Added B obj {id_b} (vox={vox_b})", flush=True)
                self.next_id += 1

        self.save_current()
        plt.close(self.fig)

    def on_key(self, event):
        if event.key in ('1', 'a', 'A'):
            self.choose_mask('A')
        elif event.key in ('2', 'b', 'B'):
            self.choose_mask('B')
        elif event.key in ('3', 'm', 'M'):
            self.choose_mask('Merge')
        elif event.key in ('4', 'n', 'N'):
            self.choose_mask('None')
        elif event.key in ('u', 'U'):
            self.undo()
        elif event.key in ('q', 'Q', 'escape'):
            self.save_and_quit()
        elif event.key == 'left':
            self.move_prev()
        elif event.key == 'right':
            self.move_next()

    def run(self):
        print(f"[START] {len(self.pairs)} pairs")
        plt.show()
        return {"pairs": len(self.pairs), "chosen_a": self.state.chosen_a,
                "chosen_b": self.state.chosen_b, "chosen_merge": self.state.chosen_merge,
                "chosen_none": self.state.chosen_none}


def review_one_file(img_path, a_path, b_path, out_path, **kwargs):
    reviewer = InstanceMaskReviewer(img_path, a_path, b_path, out_path, **kwargs)
    if len(reviewer.pairs) == 0:
        return reviewer.stats
    return reviewer.run()


def main():
    ap = argparse.ArgumentParser(description="Instance mask comparison")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--mask_a_dir", required=True)
    ap.add_argument("--mask_b_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--margin", type=int, default=15)
    ap.add_argument("--iou_threshold", type=float, default=0.1)
    ap.add_argument("--start_from", default="")
    args = ap.parse_args()
    dir_img = Path(args.images_dir)
    dir_a = Path(args.mask_a_dir)
    dir_b = Path(args.mask_b_dir)
    out_dir = Path(args.out_dir)
    for p, name in [(dir_img, "images"), (dir_a, "mask_a"), (dir_b, "mask_b")]:
        if not p.is_dir():
            print(f"[ERROR] {name}: {p}")
            sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = intersect_names(dir_img, dir_a, dir_b)
    if not names:
        print("[ERROR] No files")
        sys.exit(1)
    if args.start_from and args.start_from in names:
        names = names[names.index(args.start_from):]
    print(f"[INFO] Files: {len(names)}")
    print(f"[INFO] Output: {out_dir.resolve()}\\n")
    total = {"files": 0, "pairs": 0, "chosen_a": 0, "chosen_b": 0, "chosen_merge": 0, "chosen_none": 0}
    for i, name in enumerate(names, 1):
        print(f"\\n{'=' * 60}\\n[{i}/{len(names)}] {name}\\n{'=' * 60}")
        plt.close('all')
        time.sleep(0.05)
        try:
            stats = review_one_file(dir_img / name, dir_a / name, dir_b / name, out_dir / name,
                                    margin=args.margin, iou_threshold=args.iou_threshold)
            total["files"] += 1
            total["pairs"] += stats["pairs"]
            total["chosen_a"] += stats["chosen_a"]
            total["chosen_b"] += stats["chosen_b"]
            total["chosen_merge"] += stats["chosen_merge"]
            total["chosen_none"] += stats.get("chosen_none", 0)
        except KeyboardInterrupt:
            print("\\n[STOP]")
            break
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gc.collect()
    print(f"\\n{'=' * 60}\\n[DONE]\\n  Files: {total['files']}\\n  Pairs: {total['pairs']}\\n"
          f"  A: {total['chosen_a']}\\n  B: {total['chosen_b']}\\n  Merge: {total['chosen_merge']}\\n"
          f"  None: {total['chosen_none']}\\n{'=' * 60}")


if __name__ == "__main__":
    main()