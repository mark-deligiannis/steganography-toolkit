import argparse
from PIL import Image
import numpy as np
from os.path import splitext

parser = argparse.ArgumentParser(description="Embed or extract a secret message in an image.")
subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

# Embed mode
embed_parser = subparsers.add_parser("embed", help="Embed a secret message into a carrier image.")
embed_parser.add_argument("--carrier", required=True, help="Path to the carrier image file.")
embed_parser.add_argument("--secret", required=True, help="Path to the secret file to embed.")
embed_parser.add_argument("--output", required=True, help="Path to save the output image with embedded secret.")

# Extract mode
extract_parser = subparsers.add_parser("extract", help="Extract a secret message from an image.")
extract_parser.add_argument("--image", required=True, help="Path to the image with embedded secret.")
extract_parser.add_argument("--output", required=True, help="Path to save the extracted secret.")

args = parser.parse_args()

# Supported formats for carrier/secret/reconstructed output files
supported_formats_all = ["png","bmp","tiff","ppm","pgm","pbm","jpg","jpeg","webp"]
# Supported formats for encoded files (lossy formats are not allowed, as they interfere with the pixel LSBits which hold useful information)
supported_formats_lossless = ["png","bmp","tiff","ppm","pgm","pbm"]

def check_file_type(path,role,lossy_allowed=True):
    # Get extension
    ext = splitext(path)[-1][1:]
    # Check if extension is supported
    if ext not in (supported_formats_all if lossy_allowed else supported_formats_lossless ):
        print(f"Error! Filetype .{ext} not supported for parameter {role}")
        exit(1)

if args.mode == "embed":
    # Test file types
    check_file_type(args.carrier,"carrier",lossy_allowed=True)
    check_file_type(args.secret, "secret", lossy_allowed=True)
    check_file_type(args.output, "output", lossy_allowed=False)
    # Load carrier image
    image = np.asarray(Image.open(args.carrier),dtype=np.uint8)
    # Get dimensions and size
    i_shape, i_size = image.shape, image.size
    # Flatten to facilitate processing
    image = image.flatten()
    # Load secret image
    secret = np.asarray(Image.open(args.secret ),dtype=np.uint8)
    # Get dimensions and size
    s_shape = secret.shape
    # Account for grayscale images
    if len(s_shape) == 2: s_shape = (*s_shape,1)
    # Encode secret dimensions at beginning
    secret = np.concatenate((
        np.array((
            s_shape[0]//256,s_shape[0]%256,
            s_shape[1]//256,s_shape[1]%256,
            s_shape[2]
        ),dtype=np.uint8),
        secret.flatten()
    ))
    # Get final size
    s_size = secret.size
    # Enforce size constraints
    if i_size < s_size*8:
        print("Error! Secret too large to hide in image. Try again with smaller secret or larger image.")
        exit(1)
    print(f"Embedding secret '{args.secret}' into carrier '{args.carrier}' and saving to '{args.output}'.")
    # Expand secret
    exp_secret = np.repeat(secret,8)
    shifts = np.tile(np.arange(0,8,1,dtype=np.uint8), s_size)
    exp_secret >>= shifts
    exp_secret &= 1
    # Embed into image
    image[:s_size*8] &= 254
    image[:s_size*8] |= exp_secret
    # Save to output file
    Image.fromarray(image.reshape(i_shape)).save(args.output)

elif args.mode == "extract":
    # Test file types
    check_file_type(args.image, "image", lossy_allowed=False)
    check_file_type(args.output,"output",lossy_allowed=True )
    # Load image
    image = np.asarray(Image.open(args.image),dtype=np.uint8).flatten()
    # Get size
    i_size = image.size
    # Define decoding function
    def compact(data, size):
        shifts = np.tile(np.arange(0,8,1,dtype=np.uint8), size)
        data &= 1
        data <<= shifts
        return data.reshape((size,8)).sum(axis=1,dtype=np.uint8)
    # Extract image dimensions
    metadata = compact(image[:5*8],5)
    s_shape = (int(metadata[0])*256+int(metadata[1]),
               int(metadata[2])*256+int(metadata[3]),
               int(metadata[4]))
    s_size = s_shape[0]*s_shape[1]*s_shape[2]
    # Extract data
    try:
        data = compact(image[5*8:(5+s_size)*8], s_size).reshape(s_shape)
    except Exception as e:
        print(f"There does not seem to be any secret hiding in '{args.image}'. Terminating...")
        exit(1)
    print(f"Extracting secret from image '{args.image}' and saving to '{args.output}'.")
    # Save to output file
    Image.fromarray(data).save(args.output)

else:
    raise Exception("Invalid mode of operation!")