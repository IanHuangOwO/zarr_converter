import argparse
import logging
from utils.reader import FileReader
from utils.writer import FileWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert image volume to multiscale OME-Zarr or other formats.")
    
    # Positional arguments
    parser.add_argument("input", type=str, help="Input file or directory path")
    parser.add_argument("output", type=str, help="Output directory for result files")
    parser.add_argument("output_type", type=str, choices=["OME-Zarr", "Zarr", "Tif", "Scroll-Tif", "Nifti", "Scroll-Nifti"],
                        help="Specify the output format: OME-Zarr, Zarr, Tif, Scroll-Tif, Nifti, Scroll-Nifti.")
    
    # File Reader options
    parser.add_argument("--transpose", action="store_true",
                        help="Transpose the input volume (swap X and Y axes)")
    
    # File Writer options
    parser.add_argument("--resize-shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Override the full-resolution volume shape")
    parser.add_argument("--resize-order", type=int, default=0,
                        help="Interpolation order for resizing: 0=nearest, 1=bilinear etc.")
    
    # OME options
    parser.add_argument("--downscale-factor", type=int, default=2,
                        help="Downsampling factor per pyramid level")
    parser.add_argument("--levels", type=int, default=5,
                        help="Number of pyramid levels to generate")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="Chunk size for Zarr storage")
    
    # Scroll option
    parser.add_argument("--scroll-axis", type=int, default=0, choices=[0, 1, 2],
                        help="Axis to scroll and save 2D slices along (0=z, 1=y, 2=x). Default is 0 (z-axis).")
    
    # Memory limit
    parser.add_argument("--memory-limit", type=int, default=64,
                        help="Maximum memory (in GB) for temp buffers")

    return parser.parse_args()

def main():
    args = parse_args()

    logging.info("Starting conversion process.")
    logging.info(f"Input path: {args.input}")
    logging.info(f"Output path: {args.output}")
    logging.info(f"Output type: {args.output_type}")
    logging.info(f"Memory limit: {args.memory_limit} GB")

    reader = FileReader(
        input_path=args.input,
        memory_limit_gb=args.memory_limit,
        transpose=args.transpose
    )

    full_res_shape = args.resize_shape if args.resize_shape else reader.volume_shape
    logging.info(f"Full-resolution shape: {full_res_shape}")

    writer = FileWriter(
        reader=reader,
        output_path=args.output,
        full_res_shape=full_res_shape,
        memory_limit_gb=args.memory_limit,
    )

    if args.output_type in ['OME-Zarr', 'Zarr']:
        writer.write_zarr(
            output_type=args.output_type,
            n_level=args.levels,
            resize_factor=args.downscale_factor,
            chunk_size=args.chunk_size,
            resize_order=args.resize_order
        )
        
    elif args.output_type in ['Tif', 'Nifti']:
        writer.write_single(
            output_type=args.output_type,
            resize_order=args.resize_order
        )
        
    elif args.output_type in ['Scroll-Tif', 'Scroll-Nifti']:
        writer.write_scroll(
            output_type=args.output_type,
            resize_order=args.resize_order,
            scroll_axis=args.scroll_axis
        )
        
    else:
        logging.error(f"Unsupported output_type: {args.output_type}")

    logging.info("Conversion complete.")

if __name__ == "__main__":
    main()