# Module A results — ChildSUn, test split

Run on Kaggle (T4), 2026-10-02. Group split checksum `42e3fb2547c5` (identical to the local split).
Single training run per model (seed 42): differences of 1-3 points between models are within run-to-run noise
seen earlier and should not be ranked without more seeds.

## Leak-free (group) vs random split

| model | split | acc | macro-F1 | unsafe recall | unsafe precision | ECE raw | ECE cal | binary ECE raw | binary ECE cal | T | best epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0 | group | 0.9202 | 0.8963 | 0.9868 | 0.9375 | 0.0459 | 0.0243 | 0.0314 | 0.0217 | 1.95 | 10 |
| efficientnet_b0 | random | 0.9925 | 0.9926 | 1.0000 | 0.9923 | 0.0057 | 0.0054 | 0.0019 | 0.0019 | 1.05 | 10 |
| mobilenetv3_large_100 | group | 0.8783 | 0.8548 | 0.9711 | 0.9134 | 0.0499 | 0.0600 | 0.0341 | 0.0272 | 1.95 | 2 |
| mobilenetv3_large_100 | random | 0.9975 | 0.9965 | 1.0000 | 0.9949 | 0.0041 | 0.0037 | 0.0030 | 0.0027 | 1.29 | 18 |
| vgg19_bn | group | 0.9151 | 0.8621 | 0.9605 | 0.9311 | 0.0440 | 0.0472 | 0.0367 | 0.0202 | 2.16 | 5 |
| vgg19_bn | random | 0.9963 | 0.9960 | 1.0000 | 0.9974 | 0.0029 | 0.0044 | 0.0013 | 0.0014 | 1.10 | 15 |

ECE = top-1 calibration error over the 10 classes; binary ECE = calibration of P(unsafe), which the alert layer uses.
T = temperature fitted on the validation split.

## Per-class recall, group split

| class | EffNet-B0 | MobileNetV3 | VGG19-BN | test images |
|---|---|---|---|---|
| babyitems | 0.85 | 0.95 | 0.84 | 55 |
| ball | 1.00 | 0.87 | 1.00 | 55 |
| fork | 1.00 | 0.91 | 0.99 | 100 |
| fruit | 0.98 | 1.00 | 0.98 | 54 |
| heater | 0.80 | 0.80 | 0.83 | 30 |
| knife | 0.90 | 0.93 | 0.97 | 60 |
| scissors | 1.00 | 0.97 | 0.97 | 35 |
| socket | 0.94 | 0.90 | 0.92 | 155 |
| spoon | 0.80 | 1.00 | 0.20 | 5 |
| toy | 0.87 | 0.79 | 0.87 | 240 |

Spoon has only 5 test images (one source photo of 5 copies is 5 images), so its recall is not reliable.

## Accuracy under corruption (group split, severity 0 / 3 / 5)

| corruption | EffNet-B0 s0 / s3 / s5 | MobileNetV3 s0 / s3 / s5 | VGG19-BN s0 / s3 / s5 |
|---|---|---|---|
| gaussian_noise | 0.92 / 0.15 / 0.22 | 0.88 / 0.23 / 0.10 | 0.92 / 0.51 / 0.09 |
| iso_noise | 0.92 / 0.15 / 0.20 | 0.88 / 0.18 / 0.11 | 0.92 / 0.54 / 0.14 |
| motion_blur | 0.92 / 0.81 / 0.61 | 0.88 / 0.76 / 0.50 | 0.92 / 0.75 / 0.51 |
| low_light | 0.92 / 0.78 / 0.36 | 0.88 / 0.68 / 0.29 | 0.92 / 0.88 / 0.78 |
| occlusion | 0.92 / 0.87 / 0.80 | 0.88 / 0.76 / 0.61 | 0.92 / 0.85 / 0.76 |
| jpeg | 0.92 / 0.90 / 0.80 | 0.88 / 0.85 / 0.78 | 0.92 / 0.92 / 0.81 |

## Findings

1. **Leakage inflates the score by 7-12 points.** Every model reaches 99.3-99.8% on the random split (close to the
   99.63% reported for EfficientNet-B0 in the paper) but 87.8-92.0% on the group split. In the random split 91.8% of
   test images have an augmented copy of the same source photo in train (docs/split_report.md). Whether the paper used
   such a split is still unverified (only its abstract was read).
2. **Unsafe recall on unseen photos is 96-99%**, with 5-15 unsafe test images classified as safe.
3. **Temperature scaling improves the binary P(unsafe) calibration for all three models** (binary ECE and NLL go down),
   but top-1 10-class ECE got slightly worse on test for MobileNetV3 and VGG19 even though it improved on validation.
   The validation split also chose the checkpoint, so it is optimistic; report both numbers.
4. **Noise is the main weakness.** Gaussian/ISO noise at severity 3 drops accuracy to 15-54%. Under strong noise some
   models put almost everything in one class, so a high unsafe recall there is collapse, not robustness. Motion blur,
   JPEG and occlusion cost much less. This motivates A5 (robust augmentation) before webcam use in dim light.
5. Out-of-domain webcam accuracy (A2) is not measured yet.
