import logging
import numpy as np
import zarr
import dask.array as da
import shutil
import os

from pathlib import Path
from numcodecs import Blosc
from skimage.transform import resize
from concurrent.futures import ProcessPoolExecutor
from utils.reader import FileReader

# Set up module-level logger
logger = logging.getLogger(__name__)

def _resize_xy_worker(args):
    """
    args = (slice_xy, target_y, target_x, dtype, order)
    """
    slice_xy, ty, tx, dt, ord = args
    return resize(
        slice_xy,
        (ty, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False
    ).astype(dt)


def _resize_xz_worker(args):
    """
    args = (slice_xz, target_z, target_x, dtype, order)
    """
    slice_xz, tz, tx, dt, ord = args
    return resize(
        slice_xz,
        (tz, tx),
        order=ord,
        preserve_range=True,
        anti_aliasing=False
    ).astype(dt)


def two_pass_resize_zarr(
    input_source: FileReader,
    output_source: zarr.Group,
    level: int,
    current_shape: tuple[int,int,int],
    target_shape: tuple[int,int,int],
    dtype: np.dtype,
    order: int = 1,
    chunk_size: int = 128,
):
    """
    Resize a 3D Zarr array in two passes with an on‐disk temp.

    Args:
      input_arr: zarr.Array, shape (Z, Y, X)
      output_arr: zarr.Array, pre‐created with shape (target_z, target_y, target_x)
      temp_group: zarr.Group in which to create temp_key dataset
      temp_key: name of the temp dataset (e.g. "temp")
      target_shape: (target_z, target_y, target_x)
      dtype: output dtype
      order: interpolation order for skimage.resize
      chunk_size: slices (for XY pass) and rows (for XZ pass)
      memory_threshold_gb: unused here (always temp→zarr), but kept for signature
    """

    current_z, _, _ = current_shape
    target_z, target_y, target_x = target_shape
    
    # 1) create the on‐disk temp buffer
    temp_arr = output_source.create_dataset(
        "temp", shape=(current_z, target_y, target_x), 
        chunks=(chunk_size, chunk_size, chunk_size),
        dtype=dtype, overwrite=True,
        compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
    )
    
    def _get_z_block(z0, z1):
        if level == 0:
            return input_source.read(z_start=z0, z_end=z1)
        else:
            return output_source[str(level - 1)][z0:z1]

    # Pass 1: XY → temp_arr (unchanged)
    with ProcessPoolExecutor(max_workers=8) as exe:
        for z0 in range(0, current_z, chunk_size):
            z1 = min(z0 + chunk_size, current_z)
            block = _get_z_block(z0, z1)  # (dz, Y, X)
            args = [
                (block[i], target_y, target_x, dtype, order)
                for i in range(block.shape[0]) # type: ignore
            ]
            resized_slices = list(exe.map(_resize_xy_worker, args))
            arr = np.stack(resized_slices, axis=0)
            logger.info(f"Writing volume to temp z: {z0} - {z1}")
            
            arr_chunk = (z1 - z0, chunk_size, chunk_size)
            darr = da.from_array(arr, chunks=arr_chunk) # type: ignore
            darr.to_zarr(temp_arr, region=(
                slice(z0, z1),
                slice(0, temp_arr.shape[1]),
                slice(0, temp_arr.shape[2])
            ))

    # Pass 2: XZ → output_arr (now with threaded writes)
    with ProcessPoolExecutor(max_workers=8) as exe:
        for y0 in range(0, target_y, chunk_size):
            y1 = min(y0 + chunk_size, target_y)
            block = temp_arr[:, y0:y1, :]  # (Z, dy, X)
            args = [
                (block[:, j, :], target_z, target_x, dtype, order)
                for j in range(block.shape[1])
            ]
            resized_slices = list(exe.map(_resize_xz_worker, args))
            arr = np.stack(resized_slices, axis=1)
            logger.info(f"Writing volume to zarr y: {y0} - {y1}")
            
            arr_chunk = (chunk_size, y1 - y0, chunk_size)
            darr = da.from_array(arr, chunks=arr_chunk) # type: ignore
            darr.to_zarr(output_source[str(level)], region=(
                slice(0, output_source[str(level)].shape[0]),
                slice(y0, y1),
                slice(0, output_source[str(level)].shape[2])
            ))

    # Clean up
    logger.info(f"Cleaning temp zarr")
    del output_source["temp"]
    

def _check_memory_limit(shape, dtype, memory_limit_gb):
    """
    Check memory usage and raise error if it exceeds the limit.
    Also returns the estimated size in GB.
    """
    total_elements = np.prod(shape)
    dtype_size = np.dtype(dtype).itemsize
    total_memory_bytes = total_elements * dtype_size
    memory_limit_bytes = memory_limit_gb * (1024 ** 3)

    size_in_gb = total_memory_bytes / (1024 ** 3)

    if total_memory_bytes > memory_limit_bytes:
        raise MemoryError(
            f"Array of shape {shape} and dtype {dtype} requires "
            f"{size_in_gb:.2f} GB, which exceeds the memory limit of {memory_limit_gb:.2f} GB."
        )
    
    return size_in_gb


def _save_slice(arr, output_base, output_type):
    if output_type.lower() in ["tif", "scroll-tif"]:
        import tifffile
        logger.info(f"Writing volume to {output_base.with_suffix('.tif')}")
        print(arr.dtype)
        tifffile.imwrite(str(output_base.with_suffix(".tif")), arr, imagej=True) # type: ignore
    elif output_type.lower() in ["nifti", "scroll-nifti"]:
        import nibabel as nib
        logger.info(f"Writing volume to {output_base.with_suffix('.nii.gz')}")
        if arr.ndim == 2:
            # Convert (Y, X) → (X, Y), then add Z=1 dimension
            arr_xyz = np.transpose(arr, (1, 0))[..., np.newaxis]  # (X, Y, 1)
        elif arr.ndim == 3:
            # Convert (Z, Y, X) → (X, Y, Z)
            arr_xyz = np.transpose(arr, (2, 1, 0))  # (X, Y, Z)
        else:
            raise ValueError(f"Unsupported array shape for NIfTI: {arr.shape}")
        nifti_img = nib.Nifti1Image(arr_xyz, affine=np.eye(4))
        nib.save(nifti_img, str(output_base.with_suffix(".nii.gz")))
    else:
        raise ValueError(f"Unsupported output_type '{output_type}'. Only 'tif', 'nifti', 'scroll-tif', and 'scroll-nifti' are supported.")


class FileWriter:
    def __init__(self, reader, output_path, full_res_shape, memory_limit_gb):
        self.reader: FileReader = reader
        self.output_path = Path(output_path)
        self.full_res_shape = full_res_shape
        self.memory_limit_gb = memory_limit_gb
        
        logger.info(f"Initialized FileWriter with output: {self.output_path}")

    def write_zarr(
        self,
        output_type="OME-Zarr",
        n_level=5,
        resize_factor=2, 
        chunk_size=128,
        resize_order=1
    ):
        if output_type == "Zarr":
            n_level = 1  # Only write base level if not multiscale
            zarr_path = self.output_path / f"{self.reader.volume_name}.zarr"
        else:
            zarr_path = self.output_path / f"{self.reader.volume_name}_ome.zarr"
        
        self.n_level = n_level
        self.resize_factor = resize_factor
        self.chunk_size = chunk_size
        self.resize_order = resize_order

        logger.info(f"Starting {output_type} write process to {zarr_path}")
        
        store = zarr.DirectoryStore(zarr_path)
        group = zarr.group(store=store)

        for level in range(self.n_level):
            if str(level) in group:
                logger.info(f"Level {level} already exists, skipping.")
                continue
            self._write_ome_level(group, level=level)

        if output_type == "OME-Zarr":
            self._write_ome_metadata(group)

        logger.info(f"{output_type} write process complete.")
    
    
    def _write_ome_level(self, out_ds, level):
        if level == 0:
            current_z, current_y, current_x = self.reader.volume_shape
            target_z, target_y, target_x = self.full_res_shape
                
        else:
            prev_shape = out_ds[str(level - 1)].shape
            current_z, current_y, current_x = prev_shape
            target_z = current_z // self.resize_factor
            target_y = current_y // self.resize_factor
            target_x = current_x // self.resize_factor

        logger.info(f"Resizing level {level} from {(current_z, current_y, current_x)} to {(target_z, target_y, target_x)} with order {self.resize_order}")
        logger.info(f"Creating level {level} with shape: {(target_z, target_y, target_x)}, dtype: {self.reader.volume_dtype}, chunksize: {self.chunk_size}")
        
        out_ds.create_dataset(
            str(level), shape=(target_z, target_y, target_x), 
            chunks=(self.chunk_size, self.chunk_size, self.chunk_size),
            dtype=self.reader.volume_dtype, overwrite=True, 
            compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
        )
        
        if (current_z, current_y, current_x) == (target_z, target_y, target_x):
            for z0 in range(0, current_z, self.chunk_size):
                z1 = min(z0 + self.chunk_size, current_z)
                arr = self.reader.read(z_start=z0, z_end=z1)
                logger.info(f"Writing volume to OME y: {z0} - {z1}")
                arr_chunk = (z1 - z0, self.chunk_size, self.chunk_size)
                darr = da.from_array(arr, chunks=arr_chunk) # type: ignore
                darr.to_zarr(out_ds[str(level)], region=(
                    slice(z0, z1),
                    slice(0, arr.shape[1]),
                    slice(0, arr.shape[2])
                ))
        else:
            two_pass_resize_zarr(
                input_source=self.reader,
                output_source=out_ds,
                level=level,
                current_shape=(current_z, current_y, current_x),
                target_shape=(target_z, target_y, target_x),
                dtype=self.reader.volume_dtype,
                order=self.resize_order,
                chunk_size=self.chunk_size
            )
        

    def _write_ome_metadata(self, group):
        logger.info("Writing OME-Zarr multiscale metadata")
        datasets = []
        for level in range(self.n_level):
            scale_factor = self.resize_factor ** level
            datasets.append({
                "path": str(level),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [scale_factor] * 3
                    }
                ]
            })

        multiscales = [{
            "version": "0.4",
            "name": "image",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"}
            ],
            "datasets": datasets
        }]

        group.attrs["multiscales"] = multiscales
        
    def write_single(self, output_type, resize_order=1):
        self.resize_order = resize_order
        dtype = self.reader.volume_dtype
        shape = self.full_res_shape

        # Memory check & log info
        size_in_gb = _check_memory_limit(shape, dtype, memory_limit_gb=self.memory_limit_gb)
        logger.info(f"Target output type: {output_type}")
        logger.info(f"Target array shape: {shape}")
        logger.info(f"Target dtype: {dtype}")
        logger.info(f"Estimated memory size: {size_in_gb:.2f} GB")

        temp_zarr_path = None  # Will be set if resizing is needed

        # Load or resize volume
        if self.reader.volume_shape == self.full_res_shape:
            arr = self.reader.read(z_start=0, z_end=self.reader.volume_shape[0])
        else:
            logger.info("Performing two-pass resize to match full resolution shape")
            temp_zarr_path = self.output_path / f"{self.reader.volume_name}_temp.zarr"
            store = zarr.DirectoryStore(temp_zarr_path)
            group = zarr.group(store=store)

            group.create_dataset(
                '0', shape=self.full_res_shape, chunks=(128, 128, 128), dtype=dtype,
                compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
            )

            two_pass_resize_zarr(
                input_source=self.reader,
                output_source=group,
                level=0,
                current_shape=self.reader.volume_shape,
                target_shape=self.full_res_shape,
                dtype=dtype,
                order=self.resize_order,
                chunk_size=128,
            )
            arr = group['0'][:]

        # Write to disk
        output_base = self.output_path / self.reader.volume_name
        _save_slice(arr, output_base, output_type)

        # Optional: cleanup temp Zarr if used
        if temp_zarr_path is not None:
            logger.info(f"Cleaning up temporary Zarr directory: {temp_zarr_path}")
            shutil.rmtree(temp_zarr_path, ignore_errors=True)

        logger.info("Write process completed.")

    def write_scroll(self, output_type, resize_order=1, scroll_axis=0):
        self.resize_order = resize_order
        dtype = self.reader.volume_dtype
        shape = self.full_res_shape

        axis_char = ["z", "y", "x"][scroll_axis]
        output_base = self.output_path / f"{self.reader.volume_name}_scroll"
        os.makedirs(output_base, exist_ok=True)

        logger.info(f"Target output type: {output_type}")
        logger.info(f"Target array shape: {shape}")
        logger.info(f"Target dtype: {dtype}")
        logger.info(f"Scroll axis: {axis_char}")

        direct_write = scroll_axis == 0 and self.reader.volume_shape == self.full_res_shape

        if direct_write:
            logger.info("Volume shape matches, writing slices directly without Zarr")

            for z0 in range(0, shape[0], 128):
                z1 = min(z0 + 128, shape[0])
                arr = self.reader.read(z_start=z0, z_end=z1)

                for i in range(z1 - z0):
                    slice_2d = arr[i, :, :]
                    filename = output_base / f"{output_base.name}_{axis_char}{(z0 + i):05d}"
                    _save_slice(slice_2d.astype(np.uint16), filename, output_type)

        else:
            logger.info("Using Zarr temp storage with optional resizing")

            temp_zarr_path = self.output_path / f"{self.reader.volume_name}_temp.zarr"
            store = zarr.DirectoryStore(temp_zarr_path)
            group = zarr.group(store=store)

            group.create_dataset(
                '0', shape=shape, chunks=(128, 128, 128), dtype=dtype,
                compression=Blosc(cname='lz4', clevel=5, shuffle=Blosc.SHUFFLE)
            )

            if self.reader.volume_shape == shape:
                for z0 in range(0, shape[0], 128):
                    z1 = min(z0 + 128, shape[0])
                    arr = self.reader.read(z_start=z0, z_end=z1)
                    group['0'][z0:z1, :, :] = arr
            else:
                logger.info("Performing two-pass resize to match full resolution shape")
                two_pass_resize_zarr(
                    input_source=self.reader,
                    output_source=group,
                    level=0,
                    current_shape=self.reader.volume_shape,
                    target_shape=shape,
                    dtype=dtype,
                    order=self.resize_order,
                    chunk_size=128,
                )

            arr = group["0"]

            for idx in range(arr.shape[axis]):  # type: ignore
                if scroll_axis == 0:
                    slice_2d = arr[idx, :, :]
                elif scroll_axis == 1:
                    slice_2d = arr[:, idx, :]
                else:
                    slice_2d = arr[:, :, idx]

                filename = output_base / f"{output_base.name}_{axis_char}{idx:05d}"
                _save_slice(slice_2d, filename, output_type)

            logger.info(f"Removing temporary Zarr store at: {temp_zarr_path}")
            shutil.rmtree(temp_zarr_path, ignore_errors=True)

        logger.info("Scroll write process completed.")
