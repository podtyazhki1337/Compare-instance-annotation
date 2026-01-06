# Instance Mask Comparison Tool

Interactive tool for comparing and merging two instance segmentation masks with visual inspection.

## Features

- **Side-by-side comparison** of two instance masks
- **Z-slider** through object bounding boxes (shows only slices with actual objects)
- **Manual review of all objects** - including unpaired objects (unique to one mask)
- **Merge mode** - combine both masks into single instance
- **Real-time saving** - results saved after each decision
- **Undo support** - revert previous choices
- **Napari-compatible output** - uint16, background=0

## Installation

```bash
pip install numpy scipy matplotlib tifffile
```

## Usage

```bash
python compare_instance_masks.py \
    --images_dir /path/to/images \
    --mask_a_dir /path/to/masks_a \
    --mask_b_dir /path/to/masks_b \
    --out_dir /path/to/output \
    --margin 15 \
    --iou_threshold 0.1
```

### Controls

| Key | Action |
|-----|--------|
| `1` or `A` | Choose mask A |
| `2` or `B` | Choose mask B |
| `3` or `M` | Merge both masks |
| `4` or `N` | Skip (don't add) |
| `U` | Undo last choice |
| `Q` | Save and quit |
| `←` `→` | Navigate objects |
| Z-slider | Browse through slices |

### Interface

```
┌────────────┬────────────┐
│  Image A   │  Image B   │  ← Clean images
├────────────┼────────────┤
│  Mask A    │  Mask B    │  ← Masks with overlays (red/green)
└────────────┴────────────┘
     Z-slider (0-N slices)
  [1.A] [2.B] [3.M] [4.None] [U] [←][→] [Q]
```

## How It Works

1. **Pair matching**: Objects matched between masks using IoU threshold
2. **Visual review**: Each pair shown side-by-side with statistics
3. **Decision**: Choose A, B, merge both, or skip (none)
4. **Unpaired objects**: Objects unique to one mask shown with ⚠️UNPAIRED marker
5. **Manual control**: You decide what to do with every object
6. **Output**: Single instance mask (uint16, background=0)

### Object Pairing

- Objects with IoU ≥ threshold are paired
- Unpaired objects (unique to one mask) marked with ⚠️UNPAIRED
- All objects must be manually reviewed - choose to include or skip

### Output Format

- **File type**: TIFF (uint16)
- **Background**: 0 (transparent)
- **Objects**: Sequential IDs (1, 2, 3...)
- **Compatible**: Napari, ImageJ, QuPath

## Statistics

Final summary includes:
- **A**: Objects chosen from mask A
- **B**: Objects chosen from mask B
- **M**: Merged objects (A+B combined)
- **N**: Skipped objects (not added to result)

## Requirements

- Python 3.7+
- numpy
- scipy
- matplotlib
- tifffile

## Notes

- Z-slider shows only slices containing the object (optimized browsing)
- Results auto-saved after each decision (safe to interrupt)
- All objects require manual review - including unpaired ones
- Skipped objects (4/N) are not added to output
- Works with 3D TIFF and NPY formats

