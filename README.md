# HelmNet: Safety Helmet Image Classification

Binary image classifier for SafeGuard Corp: does a worker in the picture wear a safety helmet (`1` = With Helmet) or not (`0` = Without Helmet)? The notebook compares four models: a small CNN and three VGG-16 transfer-learning variants.

## Files

| File | Purpose |
|---|---|
| `HelmNet_Full_Code.ipynb` | The cleaned notebook (no saved outputs) |
| `requirements.txt` | Python dependencies (minimum versions) |
| `HN_images.npy` | **Not included.** Array of shape `(4125, 200, 200, 3)`, uint8 |
| `HN_labels.csv` | **Not included here.** 4,125 rows, one `label` column (3,161 ones, 964 zeros) |

## How to run

1. `pip install -r requirements.txt`
2. Put `HN_images.npy` and `HN_labels.csv` in the same folder as the notebook, or set `DATA_DIR` in the Configuration cell (or the `HN_DATA_DIR` environment variable). On Google Colab, Drive is mounted automatically.
3. Run all cells. The first run downloads the ImageNet VGG-16 weights (about 58 MB). A GPU is recommended; the VGG models are slow on CPU.

Memory: the images are about 0.5 GB as uint8 and about 2 GB once normalised copies are made.

## Models

1. **Simple CNN**: three conv layers, a small dense head.
2. **VGG-16 base**: frozen ImageNet convolutional base plus one sigmoid unit.
3. **VGG-16 base + FFNN**: as above with dense layers 128 and 64 and dropout 0.3.
4. **VGG-16 base + FFNN + augmentation**: as model 3, trained on augmented images.

All use Adam (lr 0.001), binary cross-entropy, balanced class weights, and 20 epochs. Models are ranked on the **validation** set by the F2 score of the *Without Helmet* class (recall weighted twice as heavily as precision, because missing a non-compliant worker costs more than a false alarm). Only the selected model is scored on the test set, once.

## What was changed from the original notebook

**Bugs and correctness**
- The second "visualize predictions" block for model 1 displayed validation image 33 but predicted on image 12. Fixed (one helper function now serves every model).
- Training with `shuffle=False` and `steps_per_epoch = N // batch_size` meant the batches were always in the same order and the last 20 training images were never used. Now shuffled, with all batches used.
- The test set was used to compare models (models 1 and 3 were both scored on it before one was chosen). Selection is now done on validation data only, and the test set is scored once for the chosen model.
- The conclusions hard-coded percentages (12%, 5%, 9%, 16%) that would not update when the notebook is re-run. They are replaced by computed tables and a computed test summary.
- Models 3 and 4 were compiled with the string `'adam'`, ignoring the optimizer defined just above. All models now use the same configured optimizer.
- Models 2 to 4 shared one VGG-16 object; each now gets its own frozen copy.
- Recall/precision were tracked during training for models 3 and 4 only, and for the *With Helmet* class, which confused the comparison. All models now report the same metrics.

**Evaluation**
- The micro-averaged recall, precision and F1 columns all equal accuracy for two classes. They are replaced by per-class recall, precision and F1 (plus accuracy and macro F1), centred on the *Without Helmet* class.
- Each model is predicted once per split rather than two or three times.

**Portability and hygiene**
- Removed the Colab-only dependency (`google.colab`, `cv2_imshow`) and the unused OpenCV import; Drive mounting is optional and automatic only on Colab.
- Configurable data path; shape and label sanity checks at load.
- Replaced the exact-version `pip install` (numpy 2.0.2, etc.) with `requirements.txt`.
- All settings (seeds, epochs, batch size, learning rate) are in one Configuration cell.
- Removed unused imports and the stray heading in the VGG markdown cell.

**One judgement call to review**
- Model 4's augmentation used vertical flips and rotations up to 180 degrees, which create upside-down workers that never appear at test time. It now uses horizontal flips, rotations up to 20 degrees, and small shifts/zoom. This is untested on the real images; to restore the original behaviour set `rotation_range=180` and `vertical_flip=True` in the augmentation cell.

## What was and was not verified

The 472 MB image file was not available, so **the notebook has not been run on the real data and I have no new metrics to report.**

- The notebook was executed end to end (all cells, no errors) on synthetic random images of the same shape (330 images, labels sampled from your `HN_labels.csv` with the same class ratio), with 2 epochs and without pre-trained weights. That confirms the code runs and the pieces fit together; the synthetic numbers mean nothing.
- Not tested: training with the ImageNet weights and a full 20 epochs; GPU runs.

## Results in the original saved run (for reference only)

From the saved outputs of the original notebook, not from the cleaned version: validation accuracy was about 0.915 (CNN), 0.920 (VGG base), 0.900 (VGG + FFNN) and 0.879 (VGG + FFNN + heavy augmentation). Expect the cleaned notebook's numbers to differ, particularly for model 4.

## Ideas not applied

VGG-16 was trained with inputs divided by 255, whereas its ImageNet weights expect `vgg16.preprocess_input` scaling; switching may help but needs a real run to confirm. Other options: early stopping, fine-tuning the last VGG block, a lighter backbone such as MobileNetV2, and tuning the 0.5 decision threshold to trade recall for false alarms.
