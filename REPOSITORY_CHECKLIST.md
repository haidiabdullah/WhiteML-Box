# WhiteML-Box GitHub / SoftwareX Repository Checklist

Use this before submitting the SoftwareX manuscript.

## Required before submission

- [ ] Repository name is exactly `WhiteML-Box`.
- [ ] Repository is public, unless the journal explicitly accepts a private link during review.
- [ ] `README.md` opens cleanly on GitHub.
- [ ] `LICENSE` is present and says MIT.
- [ ] `requirements.txt` is present.
- [ ] `mlbox_app.py` is present.
- [ ] `data/Spectra_Nitrogen.xlsx` is present.
- [ ] `data/WhiteML-Box.tif` is present.
- [ ] `docs/USER_GUIDE.md` is present.
- [ ] `docs/MAPPING_WORKFLOW.md` is present.
- [ ] `docs/SOFTWAREX_METADATA.md` is present.
- [ ] A GitHub release/tag named `v0.7.2` exists.
- [ ] The manuscript repository URL matches the actual GitHub URL.
- [ ] The release URL opens without a 404 error.

## Local check

From the repository root, run:

```bash
python scripts/check_repository.py
```

Expected output should report required files, Python syntax, spreadsheet dimension, and raster metadata.

## Do not upload by accident

Do not upload local virtual environments, build folders, installer outputs, cache folders, or private drafts unless you intentionally want them public.
