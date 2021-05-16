import cv2
import numpy as np


class FinderPatternDetector:
    def __init__(self):
        pass

    def detect(self, input_img_path, output_img_path, verbose=False):
        input_img = cv2.imread(str(input_img_path.absolute()), cv2.IMREAD_COLOR)
        if verbose:
            cv2.imwrite("result0.jpg", input_img)

        input_img_gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img = self.__preprocess_image(input_img_gray)
        if verbose:
            cv2.imwrite("result1.jpg", img)

        mask = self.__detect_edges(img)
        if verbose:
            cv2.imwrite("result2.jpg", mask)

        contours, hierarchy, candidates = self.__detect_contours(mask)
        if verbose:
            self.__plot_contours(input_img, contours, hierarchy, candidates)

        good_candidates = self.__filter_candidates(contours, hierarchy, candidates)
        if verbose:
            self.__plot_contours(input_img, contours, hierarchy, good_candidates)

        self.__create_output(input_img, input_img_path, output_img_path, contours, hierarchy, good_candidates)

    def __preprocess_image(self, input_image):
        img = input_image.copy()
        img = cv2.GaussianBlur(img, (9, 9), 0)

        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 2)
        img = cv2.medianBlur(img, 9)
        erosion_size = 2
        erosion_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * erosion_size + 1, 2 * erosion_size + 1),
            (erosion_size, erosion_size)
        )
        img = cv2.erode(img, erosion_element)

        return img

    def __detect_edges(self, img):
        detected_edges = cv2.Canny(img, 80, 120, 7)
        mask = (detected_edges != 0).astype(img.dtype) * 255

        dilation_size = 1
        dilation_element = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (dilation_size, dilation_size)
        )
        mask = cv2.dilate(mask, dilation_element)

        return mask

    def __detect_contours(self, mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]

        all_candidates = []
        candidates = []
        for i in range(hierarchy.shape[0]):
            all_candidates.append(i)
            child = i
            layers = 0

            while hierarchy[child, 2] != -1:
                child = hierarchy[child, 2]
                layers += 1

            if hierarchy[child, 2] != -1:
                layers += 1

            if 4 <= layers <= 7:
                candidates.append(i)

        return contours, hierarchy, candidates

    def __filter_candidates(self, contours, hierarchy, candidates):
        good_candidates = []
        good_layers = []
        for candidate in candidates:
            layers = [candidate]
            child = hierarchy[candidate, 2]
            while child != -1:
                layers.append(child)
                child = hierarchy[child, 2]

            cont_a = contours[layers[0]]
            cont_b = contours[layers[1]]
            cont_c = contours[layers[2]]

            area_ratio1 = cv2.contourArea(cont_a) / cv2.contourArea(cont_b)
            area_ratio2 = cv2.contourArea(cont_b) / cv2.contourArea(cont_c)
            target_area_ratio1 = 49 / 25
            target_area_ratio2 = 25 / 9
            approx = cv2.approxPolyDP(
                contours[candidate],
                0.05 * cv2.arcLength(contours[candidate], True),
                True
            )
            if len(approx) != 4:
                continue

            w = np.sqrt(np.sum((approx[0, 0] - approx[1, 0]) ** 2).astype(float))
            h = np.sqrt(np.sum((approx[1, 0] - approx[2, 0]) ** 2).astype(float))
            if abs(w - h) / min(w, h) > 0.25:
                continue

            if not 0.5 * target_area_ratio1 < area_ratio1 < 2 * target_area_ratio1:
                if not 0.5 * target_area_ratio2 < area_ratio2 < 2 * target_area_ratio2:
                    continue

            good_candidates.append(candidate)
            good_layers.append(layers)

        duplicates = set()
        for i, layers1 in zip(good_candidates, good_layers):
            for j, layers2 in zip(good_candidates, good_layers):
                if i == j:
                    continue
                if layers1[0] in layers2[1:]:
                    duplicates.add(i)

        final_candidates = [candidate for candidate in good_candidates if candidate not in duplicates]
        return final_candidates

    def __plot_contours(self, base_img, contours, hierarchy, candidates):
        base = base_img.copy()
        color = (255, 0, 0)
        for candidate in candidates:
            cv2.drawContours(base, contours, candidate, color, 5, cv2.LINE_8, hierarchy, 0)
        cv2.imwrite("result3.jpg", base)

    def __create_output(self, base_img, input_img_path, output_img_path, contours, hierarchy, candidates):
        base = base_img.copy()
        color = (255, 0, 0)
        for candidate in candidates:
            cv2.drawContours(base, contours, candidate, color, 5, cv2.LINE_8, hierarchy, 0)
        cv2.imwrite(str(output_img_path / input_img_path.name), base)
