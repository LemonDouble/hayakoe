# OpenJTalk Dictionary Bundling

HayaKoe's Japanese G2P (pronunciation conversion) depends on [pyopenjtalk](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt).

The original pyopenjtalk downloads its dictionary from the network on first import, but HayaKoe uses a **fork that bundles the dictionary inside the wheel**, eliminating this delay.

## The Problem

The original pyopenjtalk, at `import pyopenjtalk` time, **automatically downloads via HTTPS** the OpenJTalk Japanese dictionary (`open_jtalk_dic_utf_8-1.11`, ~23 MB) if it is not present locally.

After download, it is cached at `~/.local/share/pyopenjtalk/` and is not re-downloaded afterward.

However, Docker containers start with an empty filesystem each time, so **the download repeats every time a container is launched**.

In environments where the network is blocked, the import itself fails.

## Implementation

The custom fork ([lemon-pyopenjtalk-prebuilt](https://github.com/LemonDoubleHQ/lemon-pyopenjtalk-prebuilt)) was modified to **include the dictionary files inside the package during wheel build**.

- The build workflow downloads and extracts `open_jtalk_dic_utf_8-1.11.tar.gz`, placing it inside the `pyopenjtalk/` package
- Dictionary inclusion is declared in `pyproject.toml`'s `package-data` and `MANIFEST.in`

On the HayaKoe side, specifying `lemon-pyopenjtalk-prebuilt` as a dependency in `pyproject.toml` completes the integration.

After installation, you can verify the dictionary is included inside the package:

```
site-packages/pyopenjtalk/
  +- open_jtalk_dic_utf_8-1.11/   <- bundled dictionary
  +- openjtalk.cpython-310-x86_64-linux-gnu.so
  +- __init__.py
```

## Improvement Results

- Network call on first import completely eliminated
- Works immediately after installation in offline and air-gapped environments
- Docker image build results are identical regardless of network conditions
