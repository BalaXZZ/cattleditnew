# Cattle Identification

Muzzle-based enrollment, verification, and identification of individual cattle.
The permanent animal ID is separate from versioned biometric embeddings.

## First milestone: validate the dataset

Place each animal's images in its own folder:

    data/raw/cow_001/image_01.jpg
    data/raw/cow_001/image_02.jpg
    data/raw/cow_001/image_03.jpg
    data/raw/cow_002/image_01.jpg

Folders represent individual animals, not breeds. Session subfolders are supported.
Keep original images. Record capture sessions separately when known; do not infer
session dates from filenames. Full-body images will need muzzle annotations/crops.

Run from the project root:

    .\.venv\Scripts\python.exe scripts/check_dataset.py

The checker writes outputs/dataset_manifest.csv and outputs/dataset_report.json.
It checks organization, image counts, and exact duplicate files. It does not yet
check image decoding, muzzle visibility, labeling correctness, or near duplicates.

## Development

The initial checker environment uses Python 3.14. Select .venv/Scripts/python.exe in VS Code. Choose a supported Python version for ML dependencies when training is implemented.
The checker uses only the Python standard library. ML dependencies will be added
when we implement the training baseline and choose local GPU or Colab execution.

## Roadmap

1. Validate identity labels, image quality, and capture sessions.
2. Create identity-disjoint train/validation/test splits.
3. Train a pretrained lightweight backbone with an embedding head and ArcFace.
4. Calibrate matching thresholds on validation identities; evaluate held-out cows.
5. Add muzzle detection and capture quality checks.
6. Build enrollment, verification, and duplicate-review APIs and interface.

Never randomly split photographs of the same cow across training and evaluation.
For evaluation, use separate enrollment and query sessions where available.
Research results do not establish national-scale uniqueness or lifetime stability.

## Data handling

Images, model artifacts, and generated reports are excluded from Git.
Maintain a separate backup of the original dataset. No data is uploaded by this project.


## Image audit and provisional clean dataset

Run scripts/audit_images.py with Python plus Pillow and NumPy. This decodes images,
checks byte/pixel duplicates, computes conservative quality flags and writes contact
sheets under outputs/image_audit. Heuristic scores are review aids, not acceptance tests.

The 2026-10-01 audit decoded all 1,000 images. A separate data/clean copy contains
319 IDs and 971 images. Nine IDs are excluded pending review: cow_5, cow_6,
cow_17, cow_159, cow_199, cow_200, cow_206, cow_232, cow_233. Six have cross-ID
exact image conflicts and three have fewer than three distinct decoded images.
Original raw data is unchanged. Do not infer that excluded IDs are different animals
or merge them without trustworthy evidence. Cow IDs retain their original numbering.

The representative sample was visually inspected; all images have contact sheets,
with subsequent assistant contact-sheet review and selected full-image corrections. Framing varies, so full images should not be
used as if they were muzzle-only crops. The single dark flag (cow_302 image b)
was inspected and retained: dark background alone does not establish unusability.

outputs/image_audit/muzzle_annotations.csv now contains assistant-reviewed bounding boxes.
Enter muzzle boxes in pixels on the EXIF-oriented image, upper-left origin, with
exclusive x_max/y_max. session_id stays blank until supported by capture records.
The crop export contains 604 images from 198 cows; detection splits and training are described below.

## Local muzzle annotation tool

Start scripts/annotate_muzzles.py using Python with Pillow installed, then open
http://127.0.0.1:8765. Draw around the textured muzzle including both nostrils
and a small margin. Save each box or reject unusable captures. Previous/Next
navigation does not save a drawn box; use Save box and next. Next unreviewed
resumes saved work. Only one reviewer/browser should write annotations at a time.

Annotations persist in outputs/image_audit/muzzle_annotations.csv. Coordinates
refer to the EXIF-oriented full-resolution image. The clean photos remain unchanged.
Do not call saved crop boxes quality-certified without reviewing muzzle detail.

After all images have been reviewed, run scripts/export_muzzle_crops.py.
It validates all accepted boxes, exports original-resolution RGB PNG crops to
data/muzzle_crops, and omits cows with fewer than three accepted images.
Pending annotations stop export; existing crop folders are not overwritten.
Crops have been exported. Original photos and the pre-annotation CSV backup are retained.

Preprocessing dependencies are listed in requirements-preprocessing.txt.

## Automated muzzle proposals

scripts/auto_muzzle_proposals.py generates resumable Grounding DINO proposals locally,
using .detector-venv. Outputs are under outputs/auto_muzzles. Proposals are not reviewed
annotations and must not be passed to training as certified boxes. Model weights and
runtime downloads stay outside Git. No cattle images are uploaded.


## YOLO muzzle detector

Use `.detector-venv` (Python 3.12, CUDA PyTorch) for ML work. Install
`requirements-training.txt` without replacing the installed CUDA PyTorch.

    .detector-venv/Scripts/python.exe scripts/prepare_yolo_dataset.py
    .detector-venv/Scripts/python.exe scripts/train_muzzle_yolo.py

Dataset preparation uses all 798 accepted full images with one muzzle box each.
It does not train on cropped muzzle images. Cows with fewer than three accepted
photos remain useful for detection, although omitted from the identity crop export.
Rejected blurred images are omitted, not treated as negative examples.

`data/yolo_muzzle/manifest.json` records reproducible seed-42 cow assignments:
70% train, 15% validation, 15% test, approximately by cow count. No cow or exact
pixel duplicate crosses partitions. Preserve these assignments for future identity
evaluation so the detector has not trained on evaluation cows. Capture sessions
are unknown; these splits do not establish independence of farms or backgrounds.

YOLO11n starts from official pretrained weights with one class, muzzle. Training
uses 640-pixel inputs, batch 4, up to 60 epochs, validation early stopping, moderate
augmentation and no RAM image cache. The 4 GB GPU and 8 GB RAM limit throughput.
Resume interrupted training with `scripts/train_muzzle_yolo.py --resume`.

Weights and validation curves are under `outputs/yolo/muzzle_yolo11n`.
The best validation checkpoint is evaluated once on held-out test cows, with
results in `outputs/yolo/test_summary.json`. Detection mAP, precision and recall
measure agreement with assistant-reviewed pseudo-labels, not cow identification
accuracy. These labels need independent boundary review and field evaluation.
The dataset currently lacks verified no-muzzle/background negative examples.

Ultralytics is AGPL-3.0 licensed with an Enterprise option. For a proprietary
commercial/government deployment, resolve applicable licensing before release:
https://www.ultralytics.com/license . This local training does not purchase a license.

After training, detect and crop one new photo:

    .detector-venv/Scripts/python.exe scripts/predict_muzzle.py "path/to/photo.jpg" --output outputs/new_prediction

The prediction directory must be new. A crop is saved only when exactly one
muzzle is detected. Zero/multiple detections require another capture or review.
The confidence threshold is provisional; it is not an identity-match threshold.


The first YOLO run completed 60 epochs, selecting epoch 47. Test mAP50: 95.97%; mAP50-95: 68.41%. At fixed confidence 0.25 and IoU 0.5: precision 88.28%, recall 94.17%, with 1 no-detection and 9 multiple-detection images. See outputs/yolo/test_summary.json and fixed_threshold_test.json. These are detection metrics against reviewed labels, not identity accuracy. scripts/review_yolo_predictions.py creates the fixed-threshold report and held-out crop preview.


## Muzzle identity embeddings and local registry

Run scripts/prepare_embedding_dataset.py once to reuse detector cow partitions.
Train with scripts/train_muzzle_embeddings.py using .detector-venv. ResNet18 with
an ArcFace training head creates a normalized 256-dimensional embedding. Best
weights, calibration and evaluation are in outputs/embeddings/resnet18_arcface.
The first run completed 27 epochs, selected epoch 17, and reached 71.875% closed
rank-one identification on 31 unseen test cows (96 dependent query trials).
Conservative open-set acceptance is only 28.89%, with 0.68% unknown false acceptance
in a small 15-cow gallery. This is below the intended 95% accuracy target.
Reviewed crops were used for identity evaluation; detector errors are not included.

New cows can be enrolled without retraining. Example commands:

    .detector-venv/Scripts/python.exe scripts/cow_identity.py enroll --cow-id COW-0001 "photo1.jpg" "photo2.jpg"
    .detector-venv/Scripts/python.exe scripts/cow_identity.py identify "query.jpg"
    .detector-venv/Scripts/python.exe scripts/cow_identity.py list

Global --registry, --device and --cropped options precede enroll/identify/list.
Use --details details.json for an optional JSON object of cow details.
Two distinct enrollment photos are required; three are preferable. Full images
are cropped by YOLO; exactly one detection is required. A possible duplicate or
inconsistent enrollment photos require review. --confirm-new is an explicit
human override after verifying the photos and candidate IDs.

The local SQLite registry stores IDs, details, photo hashes/paths, input types and
model-versioned templates. No match does not prove a cow is new. Candidate matches
require review; no automatic enrollment or certified identity decision is made.
Adding more cows changes the false-match risk; thresholds were calibrated on just
13 validation gallery cows. No statewide claim is justified. Model-version migration
is not implemented; keep source photos and re-embed before using different weights.

Run scripts/test_biometrics.py for four meaningful registry/protocol/numerical checks.
No real registry was populated; smoke-check data is isolated under work/identity_smoke.


## Active reliability model (v2)

The active pointer outputs/embeddings/active_model.json selects reliability_v2;
the original resnet18_arcface baseline and its historical test results remain.
Balanced same-cow batches and supervised contrastive + ArcFace training improved
VALIDATION top-match accuracy from 74.71% to 82.76%, and correct conservative
known-cow acceptance from 33.33% to 51.11%. Unknown false acceptance remained
0.79% on the small validation protocol. These selected development metrics are
not an independent test result and do not establish 95% or statewide accuracy.
No test photographs were decoded or used in this improvement round.

The selected runtime score averages centroid and maximum-template similarities.
Matching mode, weights and calibration are versioned together. Old registry
embeddings cannot mix with new ones; use --model-dir to read a baseline registry.
Global options precede the enroll/identify/list subcommand.

Capture-quality heuristics flag small crops, weak contrast and possible blur.
Their sharpness threshold comes only from training crops. Flagged lookup captures
return capture_review_required. Quality flags did not drop validation queries or
change the reported evaluation population. Frontal pose and visible texture still
need visual checks; capture quality is not certified automatically.

Development tools: scripts/diagnose_identity_validation.py,
scripts/train_identity_reliability.py and scripts/promote_reliability_model.py.
Training experiments refuse to overwrite incomplete runs; the completed run is
preserved. scripts/test_biometrics.py now covers eight meaningful checks.


## Active CNN + ViT architecture (v3)

The active model is outputs/embeddings/cnn_vit_v3. CNN-only reliability_v2 remains
available via --model-dir. The bundle contains CNN and DINOv2-small checkpoints;
both yield normalized 256-dimensional embeddings, stored separately by segment
in each registry template. Calibrated 50/50 branch scores combine centroid and
maximum-template similarities. New cows can enroll without retraining.

The selected ViT has a learned head and low-rate fine-tuning of its last two
blocks after a frozen-backbone stage. VALIDATION top-match accuracy improved
from 82.76% to 87.36%, while correct accepted known queries improved from 51.11%
to 64.44%. Unknown false acceptance remained 0.79% on a small repeated validation
protocol. These selected development metrics are not independent test accuracy,
do not reach the 90% target and do not establish statewide error rates. No test
images were loaded in this comparison round.

src/identity_system.py loads single encoders or a versioned dual-model bundle.
The bundle hash binds both checkpoints, preprocessing and fusion weights to
calibration. Old registry embeddings cannot be mixed with the new system.
Use scripts/migrate_cow_registry.py SOURCE_DB NEW_DB to re-embed original photos
while retaining cow IDs/details and preserving the source DB. Missing/modified
photos or unknown input kinds stop migration. Review the migration quality report.
Global --registry selects the resulting database for cow_identity.py.

Training/comparison: scripts/train_cnn_vit_identity.py. Packaging:
scripts/package_cnn_vit_system.py. Completed artifacts are retained under
outputs/cnn_vit; commands refuse to overwrite the completed model bundle.
The eight existing tests and five scripts/test_cnn_vit.py checks pass.
No real registry was populated or overwritten; demo checks stay under work/.

## Local identification UI

From the project folder, run:

```powershell
.\.detector-venv\Scripts\python.exe scripts/serve_identity_ui.py --port 8766
```

Open http://127.0.0.1:8766/ in your browser. The server uses CUDA when available;
pass `--device cpu` to run on the CPU. First inference loads both models.
It binds to localhost only and processes photos on this computer.

Enroll cow: supply a permanent ID, optional owner/breed/location, and one or more
different photos of one physical cow. There is no photo-count cap; three or more
photos are recommended. Single-photo enrollment requires explicit review. Choose full photo (YOLO crop) or muzzle
close-up; use the same input type for all enrollment photos. Quality concerns,
low consistency or a possible existing cow require explicit review before saving.
Original normalized uploads are retained under data/registry/captures for later
model migration. Identification photos are temporary and removed after inference.

Identify cow: upload one photo and select its input type. The UI displays the
actual muzzle used, candidate ID/details, similarity score and calibrated threshold.
An empty registry cannot return an enrolled ID. Uncertain or poor-quality captures
are flagged; no auto-enrollment occurs. Scores are similarities, not probabilities.

Cow registry lists enrolled IDs, template counts and all uploaded photos.
The selection preview shows every photo with a remove button. Additional file
selections add to the current enrollment. Uploads are staged one photo at a time
so there is no aggregate upload-size limit for an enrollment. Each original file
must still be under 15 MB and 24 megapixels. Data persists in
data/registry/cows.sqlite3. `--registry PATH` selects a separate registry; migration
is required if an existing registry uses an older model. This prototype has no
authentication and is not a public or statewide deployment.

Ten real-model UI backend checks pass with isolated test data:
`scripts/test_identity_ui.py` tests enrollment, lookup, full-image YOLO cropping,
duplicate-photo rejection, retention of migration source photos, four-photo
enrollment/gallery retrieval, staged upload cleanup and single-photo review.

Identification privacy: below-threshold matches and captures with quality flags
return no candidate IDs or cow details. The UI prompts enrollment for no-match
results and recapture for poor quality. Only an accepted registered match
returns its cow ID/details. A no-match result cannot establish that a cow is new.

## MobileNetV3-L + CosFace controlled experiments

Completed scripts/experiment_mobilenet_cosface.py with the existing identity
splits and seed 42. Matched 224 and 384 inputs and a third 384 partial-augmentation
configuration. Validation selected checkpoints: 82.8%, 89.7% and 90.8% rank-1
respectively, compared with the active CNN + ViT's 87.4%. However, correctly
accepted known-cow queries were 44.4%, 42.2% and 44.4%, versus 64.4% for the
active model. All thresholds were calibrated on validation.

The validation-selected partial candidate was compared once against the baseline
on held-out test cows. Full-muzzle rank-1: baseline 78.1%, candidate 79.2%.
Correctly accepted known cows: baseline 46.7%, candidate 33.3%. No unknown
full-muzzle query was accepted in these small test trials; this is not a zero-FAR
guarantee. Synthetic left-half rank-1: 59.4% vs 64.6%; right-half: 67.7% vs 66.7%.
No model or registry migration was performed. The active CNN + ViT remains.

Reports/checkpoints/history are under outputs/mobilenet_cosface. The four checks
in scripts/test_mobilenet_experiment.py verify CosFace target margins, correct
half crops, query exclusion and equivalence to the existing gallery protocol.
One interrupted partial training attempt was preserved and restarted; its first
21 epoch metrics were reproduced exactly.

This is a transfer-learning adaptation, not an exact replication of the paper.
Keypoint alignment was not run because verified nostril landmark labels are not
available. Historical test use, dependent folds, a single seed and synthetic
partial crops limit conclusions. Further candidate selection must not be tuned
on these reported test outcomes.
