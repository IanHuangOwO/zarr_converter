# Zarr File Converter

This tool converts 3D image volumes into multiscale OME-Zarr or other formats such as TIF and NIfTI.

---

## Quick Start (Recommended)

### 1. Install Docker

Download and install Docker for your system:

- [Docker Desktop for Windows/macOS](https://www.docker.com/products/docker-desktop)
- For Ubuntu/Linux:

```bash
sudo apt update
sudo apt install docker.io docker-compose
sudo systemctl enable docker
sudo systemctl start docker
```

Verify Docker is working:

```bash
docker --version
docker-compose --version
```

---

### 2. Run the App

#### For Linux/macOS:

```bash
chmod +x run.sh
./run.sh
```

#### For Windows (PowerShell):

```powershell
./run.ps1
```

You’ll be prompted to enter the path to your **Zarr dataset directory**. The script will handle everything from Docker build to startup.

#### What the Script Does

- Dynamically generates a `docker-compose.yml`
- Mounts your dataset at `/workspace/datas`
- Cleans up Docker resources after exit

---

## 3. Start Converting 

```bash
python main.py <input_path> <output_path> <output_type> [options]
```

### Positional Arguments
| Argument      | Description |
|---------------|-------------|
| `input`       | Path to the input file or directory |
| `output`      | Output directory for saving results |
| `output_type` | Output format: `OME-Zarr`, `Zarr`, `Tif`, `Scroll-Tif`, `Nifti`, or `Scroll-Nifti` |

### Optional Arguments
| Option | Description |
|--------|-------------|
| `--transpose` | Transpose input volume (swap X and Y axes) |
| `--resize-shape Z Y X` | Resize to a new shape (e.g., `--resize-shape 50 512 512`) |
| `--resize-order` | Interpolation order (0=nearest, 1=bilinear; default: 0) |
| `--downscale-factor` | Downsampling factor for pyramid levels (default: 2) |
| `--levels` | Number of pyramid levels to generate (default: 5) |
| `--chunk-size` | Chunk size for Zarr storage (default: 128) |
| `--scroll-axis` | Axis to scroll for 2D slices (0=z, 1=y, 2=x; default: 0) |
| `--memory-limit` | Max memory (in GB) for processing (default: 32) |

### Example
```bash
python main.py ./input.tif ./output OME-Zarr --transpose --resize-shape 100 512 512 --levels 4 --chunk-size 128
```

---

## 📜 License

MIT License