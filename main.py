from src import DocumentWithSlides
import shutil
import argparse

parser = argparse.ArgumentParser(description='SlideX, automagically extract slides from a pdf document')
parser.add_argument('-i', '--input', help='Input file name', required=True)
parser.add_argument('-o', '--output', help='Output file name', required=True)
parser.add_argument('-r', '--resolution', help='Resolution of the output image', default=500)
parser.add_argument('-s', '--slides-per-page', help='Number of slides per page', default=2)

parser.add_argument('-c', '--use-coordinates', help='Use previously saved coordinates', action='store_true')
parser.add_argument('-x', '--xcorrection', help='X coordinate to increase or decrase in the bounding box', type=int, nargs='?', default=0)
parser.add_argument('-y', '--ycorrection', help='Y coordinate to increase or decrase in the bounding box', type=int, nargs='?', default=0)
args = parser.parse_args()

dws = DocumentWithSlides(
    document_path=args.input, 
    slides_per_page=int(args.slides_per_page), 
    zoom_factor=int(args.resolution),
    use_coordinates=(True if args.use_coordinates else False),
    coordinates_correction={ "x": int(args.xcorrection), "y": int(args.ycorrection) }
)
dws.extract_slides()
dws.save_output_document(args.output)