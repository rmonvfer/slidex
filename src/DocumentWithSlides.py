import fitz
import cv2
import io, os
import numpy as np
from PIL import Image
from roipoly import MultiRoi
from matplotlib import pyplot as plt
from typing import *
import json

class DocumentWithSlides:
    """
    Represents a document containing slides to be extracted
    """

    def __init__(self, 
        document_path, 
        coordinates_correction,
        slides_per_page=2, 
        min_roi_area=1000, 
        zoom_factor=400, 
        use_coordinates=False, 
        dump_coordinates=True ):
        """
        Constructor
            :param document_path: Path to the document to be processed
            :param slides_per_page: number of slides per page
            :param min_roi_area: minimum area of the ROI
            :param zoom_factor: zoom factor of the document
        """
        # Create a blank document
        self.output_document = fitz.open()
        
        self.jsonPath = os.path.join("./", "config", "slides_coordinates.json")
        self.document_path = document_path
        self.ZOOM_FACTOR = zoom_factor
        self.DEFAULT_DPI = 72
        self.MIN_ROI_AREA = min_roi_area
        self.SLIDES_PER_PAGE = slides_per_page
        
        self.default_matrix = fitz.Matrix(self.ZOOM_FACTOR/self.DEFAULT_DPI, self.ZOOM_FACTOR/self.DEFAULT_DPI)

        try:
            self.document = fitz.open(self.document_path)
        except Exception as e:
            raise Exception("Error opening document: " + str(e))

        # Extract document pages
        self.pdf_pages = [page for page in self.document]
        self.img_pages = [page.get_pixmap(matrix=self.default_matrix) for page in self.pdf_pages]

        if use_coordinates:
            # Load the coordinates from the JSON file
            with open(self.jsonPath, "r") as f:
                self.slides_coordinates = json.load(f)
        else:
            # Extract slides coordinates
            self.slides_coordinates = self.__extract_slides_coordinates()

        if dump_coordinates:
            # Dump the coordinates to a JSON file
            with open("./config/slides_coordinates.json", "w") as f:
                json.dump(self.slides_coordinates, f)
            
        # Apply the correction to the coordinates
        if coordinates_correction:
            for idx, slide in enumerate(self.slides_coordinates):
                x, y, w, h = slide
                self.slides_coordinates[idx] = (x + coordinates_correction["x"], y + coordinates_correction["y"], w, h)
        
        x0, x1, self.slide_width, self.slide_height = self.slides_coordinates[0]
        
        # Fill the output document with blank pages
        for page_number in range(self.SLIDES_PER_PAGE * len(self.pdf_pages)):
            self.output_document.new_page(page_number, width=self.slide_width, height=self.slide_height)

    def __extract_slides_coordinates(self) -> List[Tuple[int, int, int, int]]:
        """
        Extracts the bounding boxes of the slides in the first page
        of the document. 
        """
        # Regions of interest for both slides in the first page
        slides_rois = []

        # Extract the first page
        first_page = self.img_pages[0]

        # Create a pillow Image object using first_page bytes
        img = Image.frombytes("RGB", [first_page.width, first_page.height], first_page.samples)

        # Convert the image into a Numpy Array so we can work with it
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Copy for later use
        original = img_np.copy()

        # Define masks and thresholds
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh  = cv2.threshold(blurred, 230,255, cv2.THRESH_BINARY_INV)[1]

        # Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        # Now loop through each contour and extract its bounding box
        for contour in cnts[::-1]: # Reverse order to get the biggest contour first
            if cv2.contourArea(contour) > self.MIN_ROI_AREA: # Avoid small contours
                x,y,w,h = cv2.boundingRect(contour)
                slides_rois.append((x, y, w, h))
        
        if len(slides_rois) != self.SLIDES_PER_PAGE:
            raise Exception(f"Could not find {self.SLIDES_PER_PAGE} slides in the first page!")
        else:
            return slides_rois
    
    def __pixmap_to_pil_image(self, pixmap: fitz.Pixmap) -> Image:
        """
        Converts a fitz.Pixmap object to a PIL Image object
            :param pixmap: fitz.Pixmap object
        """
        return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        
    def __crop_pdf_page(self, pagenum: int, slide_rect: Tuple[int, int, int, int]) -> Image:
        """
        Crops a given page of the document to the given ROI
        returns a PIL Image object of the cropped page
            :param pagenum: number of the page to be cropped
            :param slide_rect: coordinates of the ROI
        """
        return self.__pixmap_to_pil_image(self.img_pages[pagenum]).crop(slide_rect)
    
    def __pil_image_to_bytearray(self, pil_image: Image) -> bytes:
        """
        Converts a PIL Image object to a bytearray
            :param pil_image: PIL Image object
        """
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG")
        return img_bytes.getvalue()
    
    def __add_slide_to_pdf(self, slide_img, slide_num) -> None:
        """
        Adds a slide to the output document
            :param slide_img: PIL Image object of the slide
            :param slide_num: number of the slide
        """
        self.output_document[slide_num].insert_image(
            fitz.Rect(0, 0, self.slide_width, self.slide_height),  # Insert the image at the top left corner
            stream=self.__pil_image_to_bytearray(slide_img)       # Convert PIL Image to bytearray
        )
        
    def extract_slides(self) -> None:
        """
        Extracts slides from the loaded document
        """
        page_num = 0
        for page_number in range(len(self.pdf_pages)):
            for region in self.slides_coordinates:
                x0, y0, w, h = region
                slide_img = self.__crop_pdf_page(page_number, (x0, y0, x0+w, y0+h))
                self.__add_slide_to_pdf(slide_img, page_num)
                page_num += 1
        
    def save_output_document(self, output_path) -> None:
        """
        Saves the output document
            :param output_path: path to the output document
        """
        self.output_document.save(output_path)