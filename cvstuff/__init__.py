import cv2
import numpy as np

DEBUG = True

def exit():
    if DEBUG:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        sys.exit()

class Rectangle:
    def __init__(self, tl=0, tr=0, br=0, bl=0):
        self.tl = tl
        self.tr = tr
        self.br = br
        self.bl = bl

    def top_width(self):
        return np.linalg.norm(self.tr - self.tl)

    def bottom_width(self):
        return np.linalg.norm(self.br - self.bl)

    def left_height(self):
        return np.linalg.norm(self.tl - self.bl)

    def right_height(self):
        return np.linalg.norm(self.tr - self.br)

    def equiv_height(self):
        lh = self.left_height()
        rh = self.right_height()

        if lh > rh:
            return rh
        else:
            return lh

    def equiv_width(self):
        tw = self.top_width()
        bw = self.bottom_width()

        if tw > bw:
            return bw
        else:
            return tw

    def convert_to_np(self):
        return np.array([ self.tl, self.tr, self.br, self.bl ], np.float32)

    def equiv_area(self):
        return self.equiv_width() * self.equiv_height()

class Cell:
    def __init__(self, tl, height, img_data, centre=(0,0), num_white=0, tr=(0,0)):
        self.tl = tl
        self.height = height
        self.img_data = img_data
        self.num_white = num_white
        self.centre = centre
        self.tr = tr

    def __cmp__(self, other):
        if other.tl[1] > (self.tl[1] + self.height/2):
            return -1
        # Otherwise they are on the same row, so the smaller x value will be the smallest
        elif self.tl[0] < other.tl[0]:
            return -1
        else:
            return 1

class CrosswordRecogniser:
    GAUSSIAN_BLUR_SIZE = 5
    APPROX_POLY_TOLERANCE = 0.05
    BOX_SIZE_OFFSET = 10
    BOX_SIZE_DIVISOR_MIN = 500
    BOX_SIZE_DIVISOR_MAX = 4

    def read_image(self, name):
        return cv2.imread(name)

    def image_area(self, img):
        height, width = img.shape
        return height * width

    def scale(self, img, new_height = 1500): # CONSTANT!
        height, width, channels = img.shape
        ratio = float(width) / float(height)
        new_width = int(ratio * new_height)
        return cv2.resize(img, (new_width, new_height))

    def display_image(self, name, img):
        if DEBUG:
            cv2.imshow(name, img)

    # Expects a greyscale image
    def convert_to_greyscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def blur(self, img):
        return cv2.GaussianBlur(img, (self.GAUSSIAN_BLUR_SIZE, self.GAUSSIAN_BLUR_SIZE), 0)

    def otsu_threshold(self, img):
        _, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        return img_threshold

    def adaptive_threshold(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.GAUSSIAN_BLUR_SIZE, 2)

    # NOTE: img may be modified
    def find_contours(self, img, mode=cv2.RETR_LIST):
        # TODO: Could RETR_TREE or RETR_LIST, need to see how much perf difference it makes
        contours, hierarchy = cv2.findContours(img, mode, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # NOTE: copy of img is made, which is slow
    def draw_contours(self, img, contours):
        if not DEBUG:
            return
        dest_img = np.copy(img)
        cv2.drawContours(dest_img, contours, -1, (0,255,0), 3)
        return dest_img

    def sort_contours_by_size(self, contours):
        result = sorted(contours, key=cv2.contourArea, reverse=True)
        return result

    def approximate_contour(self, contour):
        perimeter = cv2.arcLength(contour, True)
        approx_poly = cv2.approxPolyDP(contour, perimeter * self.APPROX_POLY_TOLERANCE, True)
        return approx_poly.reshape(len(approx_poly), 2)

    # NOTE: copy of img is made, which is slow
    def draw_poly_dots(self, img, approx_poly):
        if not DEBUG:
            return
        dest_img = np.copy(img)
        for coords in approx_poly:
            cv2.circle(dest_img, (coords[0], coords[1]), 4, (0,0,255), -1)
        return dest_img

    # approx_box must come from approximate_contour
    def get_ordered_rect_from_poly(self, approx_box, use_offset=True):
        offset = 0
        if use_offset:
            offset = self.BOX_SIZE_OFFSET

        # First let's get the sums of x+y, TL is smallest, BR is biggest
        s = approx_box.sum(axis = 1)
        tl = approx_box[np.argmin(s)] + (-offset, -offset)
        br = approx_box[np.argmax(s)] + (offset, offset)

        # Then we'll do the difference (y-x), TR is smallest, BL is biggest
        d = np.diff(approx_box, axis = 1)
        tr = approx_box[np.argmin(d)] + (offset, -offset)
        bl = approx_box[np.argmax(d)] + (-offset, offset)

        return Rectangle(tl, tr, br, bl)

    def warp_to_rectangle(self, img, ordered_rect):
        width = int(ordered_rect.equiv_width())
        height = int(ordered_rect.equiv_height())
        src_rect = ordered_rect.convert_to_np()

        dst_rect = np.array([ [0,0], [width,0], [width,height], [0,height] ], np.float32)

        transform = cv2.getPerspectiveTransform(src_rect, dst_rect)
        img_warped = cv2.warpPerspective(img, transform, (width,height))

        return img_warped

    def find_potential_boxes(self, img, div_min=BOX_SIZE_DIVISOR_MIN, div_max=BOX_SIZE_DIVISOR_MAX):
        contours = self.find_contours(img)
        image_area = self.image_area(img)

        # Find contours that aren't too big or small
        potential_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > image_area/div_min and area < image_area/div_max:
                potential_boxes.append(contour)

        return potential_boxes

    def compute_contour_centre(self, cnt):
        mom = cv2.moments(cnt)
        (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
        return (x,y)

    def draw_contour_moments(self, img, contours):
        if not DEBUG:
            return
        for cnt in contours:
            mom = cv2.moments(cnt)
            centre = self.compute_contour_centre(cnt)
            cv2.circle(img, centre, 4, (0,0,255), -1)

    def img_data_top_half(self, img):
        return img[2:len(img)/2-2,2:-2]

    def get_top_half_box(self, img, contour):
        approx_box = self.approximate_contour(contour)
        ordered_rect = self.get_ordered_rect_from_poly(approx_box, False)
        img_data_box = self.warp_to_rectangle(img, ordered_rect)
        return ordered_rect, self.img_data_top_half(img_data_box)

    def train_knn(self, img, boxes):
        cells = []
        smallest_area = 999999
        width_height = (0,0)

        for cnt in boxes:
            # TODO: Maybe transform into an actual rectangle first
            rect, img_data_box = self.get_top_half_box(img, cnt)
            cell = Cell(rect.tl, rect.equiv_height(), img_data_box)
            cells.append(cell)

            # Compute the smallest area we have
            area = self.image_area(cell.img_data)
            if area  < smallest_area:
                smallest_area = area
                height, width = cell.img_data.shape
                width_height = (width*8, height*8)

        sorted_cells = sorted(cells)
        samples = np.empty((0, width_height[0] * width_height[1]))

        for i in range(len(sorted_cells)):
            cell = sorted_cells[i]
            img_data_resized = cv2.resize(cell.img_data, width_height)
            sample = img_data_resized.reshape((1, width_height[0] * width_height[1]))
            samples = np.append(samples,sample,0)

            # TODO: Remove this drawing
            #cr.display_image('blahblah', img_data_resized)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        numbers = np.arange(26, dtype=np.float32)
        samples = samples.astype(np.float32)

        model = cv2.KNearest()
        model.train(samples, numbers)

        return model, sorted_cells, width_height

    def process_crossword_contours(self, img_threshold, img_greyscale, contours):
        widths = np.zeros(len(contours))
        heights = np.zeros(len(contours))
        i = 0
        cells = []

        for cnt in contours:
            approx_poly = self.approximate_contour(cnt)
            if len(approx_poly) == 4:
                ordered_box = self.get_ordered_rect_from_poly(approx_poly, False)

                img_data_box_thresh = self.warp_to_rectangle(img_threshold, ordered_box)
                img_data_box_grey = self.warp_to_rectangle(img_greyscale, ordered_box)
                num_white = (img_data_box_grey > 70).sum() # TODO: Constant! (Check if white)

                cell = Cell(ordered_box.tl, ordered_box.equiv_height(), img_data_box_thresh, self.compute_contour_centre(cnt), num_white, ordered_box.tr)

                if num_white > 1000: # TODO: Constant! (It's a white square)
                    cells.append(cell)
                    widths[i] = ordered_box.equiv_width()
                    heights[i] = ordered_box.equiv_height()
                    i += 1

        median_width = np.median(widths[0:i])
        median_height = np.median(heights[0:i])
        sorted_cells = sorted(cells)

        return median_height, median_width, img_threshold.shape[0], img_threshold.shape[1], sorted_cells

    def generate_structure(self, mh, mw, h, w, sorted_cells):
        structure = []
        current_row = []

        start_cell = sorted_cells[0]
        diff_start = start_cell.tl[0] - self.BOX_SIZE_OFFSET
        num_black_start = int(round(diff_start/mw))

        for i in range(num_black_start):
            current_row.append(None)

        current_row.append(start_cell)

        for i in range(len(sorted_cells)-1):
            current_cell = sorted_cells[i]
            next_cell = sorted_cells[i+1]

            # x value got smaller! must be a new row
            if next_cell.tl[0] < current_cell.tl[0]:
                # Need to work out if there were any black boxes to the right of the current box
                diff_end = w - self.BOX_SIZE_OFFSET - current_cell.tl[0]
                num_black_end = int(round(diff_end/mw) - 1)

                for i in range(num_black_end):
                    current_row.append(None)

                #print len(current_row)
                #if len(current_row) != 15:
                #    print current_row

                structure.append(current_row)

                # Need to work out how many black boxes there are to the left on the new row
                diff_start = next_cell.tl[0] - self.BOX_SIZE_OFFSET
                num_black_start = int(round(diff_start/mw))

                current_row = []

                for i in range(num_black_start):
                    current_row.append(None)

                current_row.append(next_cell)
            elif next_cell.tl[0] > (current_cell.tl[0] + mw*1.5): # TODO: CONSTANT
                diff = next_cell.tl[0] - current_cell.tl[0]
                num_black = int(round(diff/mw - 1))

                for i in range(num_black):
                    current_row.append(None)

                current_row.append(next_cell)
            else:
                current_row.append(next_cell)

        diff_end = w - next_cell.tl[0]
        num_black_end = int(round(diff_end/mw) - 1)

        for i in range(num_black_end):
            current_row.append(None)

        structure.append(current_row)

        return structure

class CrosswordRecogniserInterface:
    def __init__(self):
        self.cr = CrosswordRecogniser()

    def exit(self):
        exit()

    def solve_image(self, image_path):
        # Read image
        img = self.cr.scale(self.cr.read_image(image_path))
        self.cr.display_image('Orignal Image', img)

        # Convert to greyscale
        img_greyscale = self.cr.convert_to_greyscale(img)
        self.cr.display_image('Greyscale Image', img_greyscale)

        # Blur the image
        img_blurred = self.cr.blur(img_greyscale)
        self.cr.display_image('Blurred Image', img_blurred)

        # Apply thresholding
        # TODO: Figure out if the blurring is really necessary here
        #img_threshold = cr.otsu_threshold(img_greyscale)
        img_threshold = self.cr.adaptive_threshold(img_blurred)
        self.cr.display_image('Thresholded Image', img_threshold)

        # Find and draw contours
        contours = self.cr.find_contours(np.copy(img_threshold), cv2.RETR_EXTERNAL)
        self.cr.display_image('First pass contours', self.cr.draw_contours(img, contours))

        # Sort contours and display the largest 2
        sorted_contours = self.cr.sort_contours_by_size(contours)
        self.cr.display_image('Largest contour', self.cr.draw_contours(img, sorted_contours[0:2]))

        # Start processing the largest contour
        largest_contour = sorted_contours[0]
        approx_box = self.cr.approximate_contour(largest_contour)

        # TODO: NEED TO ALLOW NON SQUARE
        if len(approx_box) != 4:
            raise Exception('Approximate box is not a box')

        self.cr.display_image('Crossword Poly Dots', self.cr.draw_poly_dots(img, approx_box))

        # Create a rectangle in the correct order so we can transform it
        ordered_rect = self.cr.get_ordered_rect_from_poly(approx_box)

        # Warp the crossword box
        img_warped = self.cr.warp_to_rectangle(img_threshold, ordered_rect)
        img_warped_crossword_copy = np.copy(img_warped)
        img_warped_color = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR)
        self.cr.display_image('Warped crossword', img_warped)

        # Find the individual boxes
        potential_boxes = self.cr.find_potential_boxes(img_warped)

        # Draw circles in centre
        self.cr.draw_contour_moments(img_warped_color, potential_boxes)

        self.cr.display_image('Crossword boxes', self.cr.draw_contours(img_warped_color, potential_boxes))

        # Get the warped greyscale version
        img_grey_warped = self.cr.warp_to_rectangle(img_greyscale, ordered_rect)
        self.cr.display_image('Warped original crossword', img_grey_warped)

        mh, mw, h, w, sorted_cells = self.cr.process_crossword_contours(img_warped_crossword_copy, img_grey_warped, potential_boxes)
        structure = self.cr.generate_structure(mh, mw, h, w, sorted_cells)

        if len(structure) != 19:
            raise Exception('Crossword structure is not correct (number of rows)')

        for row in structure:
            if len(row) != 15:
                raise Exception('Crossword structure is not correct (number of columns)')

        # Start processing the second largest contour
        second_contour = sorted_contours[1]
        approx_box = self.cr.approximate_contour(second_contour)

        # TODO: NEED TO ALLOW NON SQUARE
        if len(approx_box) != 4:
            raise Exception('Approximate box is not a box')

        self.cr.display_image('Solutions Poly Dots', self.cr.draw_poly_dots(img, approx_box))

        # Create a rectangle in the correct order so we can transform it
        ordered_rect = self.cr.get_ordered_rect_from_poly(approx_box)

        # Warp the crossword box
        img_warped = self.cr.warp_to_rectangle(img_threshold, ordered_rect)
        img_warped_copy = np.copy(img_warped)
        img_warped_color = cv2.cvtColor(img_warped, cv2.COLOR_GRAY2BGR)
        self.cr.display_image('Warped solutions', img_warped)

        # Get solution boxes
        potential_solution_boxes = self.cr.find_potential_boxes(img_warped, 52)

        # TODO: If there are not 26 of these, we've got problems
        if len(potential_solution_boxes) != 26:
            raise Exception('Too many solution boxes!')

        # Draw circles
        self.cr.draw_contour_moments(img_warped_color, potential_solution_boxes)
        self.cr.display_image('Solutions boxes', self.cr.draw_contours(img_warped_color, potential_solution_boxes))

        # Train the knn model with the solution boxes
        model, sorted_cells, width_height = self.cr.train_knn(img_warped_copy, potential_solution_boxes)

        # Foreach row in the structure, then foreach column, detect the number
        distances = []

        for r in range(len(structure)):
            row = structure[r]
            for c in range(len(row)):
                cell = row[c]

                if cell is not None:
                    img_data_top = self.cr.img_data_top_half(cell.img_data)
                    img_data_top = cv2.resize(img_data_top, width_height)
                    input_sample = img_data_top.reshape((1, self.cr.image_area(img_data_top))).astype(np.float32)

                    retval, results, neighborResponses, dists = model.find_nearest(input_sample, k=5)
                    #print neighborResponses
                    number = int(neighborResponses[0][0] + 1)
                    row[c] = number
                else:
                    row[c] = -1

        return structure
