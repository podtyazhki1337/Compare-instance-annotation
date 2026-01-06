\# Instance Mask Comparison Tool



Interactive tool for comparing and merging two instance segmentation masks with visual inspection.



\## Features



\- \*\*Side-by-side comparison\*\* of two instance masks

\- \*\*Z-slider\*\* through object bounding boxes (shows only slices with actual objects)

\- \*\*Automatic unpaired object handling\*\* - objects unique to either mask are preserved

\- \*\*Merge mode\*\* - combine both masks into single instance

\- \*\*Real-time saving\*\* - results saved after each decision

\- \*\*Undo support\*\* - revert previous choices

\- \*\*Napari-compatible output\*\* - uint16, background=0



\## Installation



```bash

pip install numpy scipy matplotlib tifffile

```



\## Usage



```bash

python compare\_instance\_masks.py \\

&nbsp;   --images\_dir /path/to/images \\

&nbsp;   --mask\_a\_dir /path/to/masks\_a \\

&nbsp;   --mask\_b\_dir /path/to/masks\_b \\

&nbsp;   --out\_dir /path/to/output \\

&nbsp;   --margin 15 \\

&nbsp;   --iou\_threshold 0.1

```



\### Controls



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



\### Interface



```

┌────────────┬────────────┐

│  Image A   │  Image B   │  ← Clean images

├────────────┼────────────┤

│  Mask A    │  Mask B    │  ← Masks with overlays (red/green)

└────────────┴────────────┘

&nbsp;    Z-slider (0-N slices)

&nbsp; \[1.A] \[2.B] \[3.M] \[4.None] \[U] \[←]\[→] \[Q]

```



\## How It Works



1\. \*\*Pair matching\*\*: Objects matched between masks using IoU threshold

2\. \*\*Visual review\*\*: Each pair shown side-by-side with statistics

3\. \*\*Decision\*\*: Choose A, B, merge both, or skip

4\. \*\*Auto-complete\*\*: Unpaired objects automatically added at the end

5\. \*\*Output\*\*: Single instance mask (uint16, background=0)



\### Object Pairing



\- Objects with IoU ≥ threshold are paired

\- Unpaired objects (unique to one mask) marked with ⚠️UNPAIRED

\- All differences are shown for review



\### Output Format



\- \*\*File type\*\*: TIFF (uint16)

\- \*\*Background\*\*: 0 (transparent)

\- \*\*Objects\*\*: Sequential IDs (1, 2, 3...)

\- \*\*Compatible\*\*: Napari, ImageJ, QuPath





\## Statistics



Final summary includes:

\- \*\*A\*\*: Objects chosen from mask A

\- \*\*B\*\*: Objects chosen from mask B

\- \*\*M\*\*: Merged objects (A+B combined)

\- \*\*N\*\*: Skipped objects

\- \*\*Auto\*\*: Automatically added unpaired objects



\## Requirements



\- Python 3.7+

\- numpy

\- scipy

\- matplotlib

\- tifffile



\## Notes



\- Z-slider shows only slices containing the object (optimized browsing)

\- Results auto-saved after each decision (safe to interrupt)

\- Unpaired objects preserved automatically (no data loss)

\- Works with 3D TIFF and NPY formats



