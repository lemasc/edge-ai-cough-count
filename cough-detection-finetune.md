The paper **"Cough-E: A multimodal, privacy-preserving cough detection algorithm for the edge"** contains a post-processing step that is _specifically designed_ to fix the two problems we're describing:

1. **Overlapping sliding windows can "double count" the same cough**, and
2. **Multiple coughs inside one merged detection can get collapsed into one long event**, hurting cough counting.

Below is what they do and how us we adapt it to our Gradio app.

---

## What the paper's post-processing does (key idea)

Their model produces **cough-positive regions at the window level**, which is not the same as **counting discrete cough events**. So when any window is classified as cough, they trigger an **audio-only refinement algorithm** to estimate cough **start/end/peak** times and then split/merge events using physiology-inspired timing rules.

### Step 1 — Within each cough-positive segment: get rough "bursts" with hysteresis on signal power

They:

- **Downsample the audio segment to 2 kHz** (cheaper, still enough for an energy envelope).
- Compute **signal power** across the segment and apply **hysteresis thresholding**:
  - **Upper threshold** = halfway between **RMS power** and **max power**
  - **Lower threshold** = **RMS power**

- When power crosses upper threshold → mark a region start; when it drops below lower threshold → mark region end
- Define a **peak** as the maximum power inside each region

This gives us lists of `(start_i, end_i, peak_time_i, peak_amp_i)` for the segment.

### Step 2 — Fix overlap/double-counting and voiced-phase "extra peaks" by peak de-duplication (tcoughDur_min)

They explicitly call out our exact issue: **window overlaps can count the same cough twice**.

They compare _adjacent peaks_ and say:

- If two peaks are closer than **tcoughDur_min = 0.23 s**, treat them as **the same cough** and:
  - keep the **larger-amplitude peak**
  - take the **union** of the two regions

That 0.23s is chosen from physiology: minimum spike (0.03–0.05s) + minimum expiratory duration (~0.2s).

### Step 3 — Split coughs _inside bouts_ using a second timing threshold (tcoughDur_max)

They define cough _bouts_ as successive coughs without pause, and use:

- **tcoughDur_max = 0.55 s** (max spike+expiration sum) to decide whether coughs are "in a bout"

In their refinement loop:

- If the next peak is within **tcoughDur_max**, they treat the current cough as part of a bout and set:
  - `end[n] = start[n+1]` (i.e., **hard split between coughs**)

This is the piece that helps us **separate coughs that our current gap-based merging collapses**.

### Step 4 — Make end-times more realistic for the _last cough in a bout_ (or a single cough)

They note that ends are hard because amplitude decays gradually. So they compute a **subject-specific average "peak-to-end" time** `t_pkToEnd_avg` to estimate the tail.

Then for a cough that is **not followed by another cough within tcoughDur_max**, they set:

- `end[n] = peak[n] + t_pkToEnd_avg * C`
  where **C decreases exponentially** as more coughs occur in the bout (coughs shorten as lungs empty).

**Important note:** the excerpt explains what C represents but does not give the explicit formula in the visible text us provided. So us'll either need to (a) check their open-source implementation, or (b) implement our own decreasing factor (e.g., `C = exp(-k * cough_index_in_bout)` and tune `k`).

---

## How this maps onto our current Gradio app (and why our merging fails)

our current pipeline merges window detections by a **gap_threshold = 0.3s** between consecutive window-level detections.

That logic is great for collapsing many overlapping windows into one event, but it has a blind spot:

- **Two real coughs close together** (especially in a bout) can produce _continuous above-threshold windows_, so our merger returns **one long event**, so our cough count becomes **1 instead of 2+**.

The paper"s fix is: **don't count merged windows; count peaks (events) inside them**, and split them using physiology-tuned timing thresholds (0.23s and 0.55s).

---

## Practical "drop-in" adaptation for our app

We don't need to change our classifier. Add a second stage **after** us identify cough-positive windows:

1. **Group cough-positive windows** into coarse "candidate segments" (us can keep our merging, but treat output as _regions to refine_, not final coughs).
2. For each candidate segment:
   - Extract the raw audio for that time span (optionally pad ±0.2s).
   - Downsample to 2 kHz.
   - Compute power envelope; run hysteresis to get `(start,end,peak)` candidates.
   - Run the paper's refinement rules:
     - **Deduplicate peaks** closer than **0.23s** (union regions, keep max peak).
     - If consecutive peaks are within **0.55s**, split by setting `end[n]=start[n+1]`.
     - Otherwise estimate the tail using `t_pkToEnd_avg * C`.

3. our final cough count = **number of refined peaks/events**, not number of merged regions.

If us want the _minimum viable_ improvement (without the `t_pkToEnd_avg * C` complexity), us can still get most of the benefit by doing just:

- hysteresis peak extraction
- **peak dedupe @ 0.23s**
- **bout split @ 0.55s**

That alone usually fixes "merged windows collapse multiple coughs".

---

## One extra tip: our current `gap_threshold=0.3s` collides with their physiology thresholds

We're merging events with gaps ≤ 0.3s.
But their "same cough" dedupe threshold is **0.23s** and their "bout" boundary logic is keyed off **0.55s**.

So it's very plausible that 0.3s merging is **too aggressive** for cough counting unless us add a peak-based splitter afterward (as above).
