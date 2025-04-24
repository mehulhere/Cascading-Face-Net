# Pending Fixes for LFW Accuracy Reproduction

Below are the key discrepancies we addressed, and what remains:

## Completed

- [x] 1. Whitening Consistency (Facenet)
- [x] 2. MTCNN Selection Method (Facenet)
- [x] 3. Margin and Resize (Facenet)
- [x] 4. Two-Phase Pipeline (Facenet)
- [x] 5. EdgeFace Preprocessing Check (Reverted to on-the-fly)
- [x] 6. Batch vs. Single-Image Norm (EdgeFace - Reverted to on-the-fly)

## Remaining

7. Threshold Search & Calibration
   - Validate fold definitions, search granularity, and consider caching embeddings.

---

Next up: Check **7. Threshold Search & Calibration**. 